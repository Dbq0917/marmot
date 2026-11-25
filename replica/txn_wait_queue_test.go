package replica

import (
	"context"
	"math/rand/v2"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// verifyInvariants checks the min-heap ordering and that every waiter's
// channel maps back to its actual position.
func verifyInvariants(t *testing.T, q *TxnWaitQueue) {
	t.Helper()
	q.mu.Lock()
	defer q.mu.Unlock()

	if len(q.index) != len(q.waiters) {
		t.Fatalf("index size %d does not match heap size %d", len(q.index), len(q.waiters))
	}
	for pos, w := range q.waiters {
		if got, ok := q.index[w.ch]; !ok || got != pos {
			t.Fatalf("waiter at %d has stale/missing index entry: got %d ok=%v", pos, got, ok)
		}
		if parent := (pos - 1) / 2; pos > 0 && q.waiters[pos].txnID < q.waiters[parent].txnID {
			t.Fatalf("heap property violated at %d (txnID %d < parent txnID %d)",
				pos, q.waiters[pos].txnID, q.waiters[parent].txnID)
		}
	}
}

func TestTxnWaitQueue_HeapAndIndexInvariants(t *testing.T) {
	q := NewTxnWaitQueue()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	const n = 500
	done := make(chan error, n)
	txnIDs := rand.Perm(n)
	for _, id := range txnIDs {
		go func(txnID int) {
			done <- q.Wait(ctx, uint64(txnID))
		}(id)
	}
	for q.Len() != n {
		time.Sleep(time.Microsecond)
	}
	verifyInvariants(t, q)

	// Waking in arbitrary batches must keep the remaining heap valid.
	for threshold := uint64(50); ; threshold += 50 {
		q.NotifyUpTo(threshold)
		verifyInvariants(t, q)
		if threshold >= n {
			break
		}
	}

	if q.Len() != 0 {
		t.Fatalf("expected empty queue, got %d", q.Len())
	}
}

func TestTxnWaitQueue_CancelOneOfDuplicateTxnIDs(t *testing.T) {
	q := NewTxnWaitQueue()

	ctx1, cancel1 := context.WithCancel(context.Background())
	ctx2, cancel2 := context.WithCancel(context.Background())
	ctx3, cancel3 := context.WithCancel(context.Background())
	defer cancel3()

	done := make(chan error, 3)
	start := func(ctx context.Context) {
		go func() { done <- q.Wait(ctx, 100) }()
	}
	start(ctx1)
	start(ctx2)
	start(ctx3)
	for q.Len() != 3 {
		time.Sleep(time.Microsecond)
	}

	// Cancel only the second waiter; the other two must remain queued.
	cancel2()
	select {
	case err := <-done:
		if err != context.Canceled {
			t.Fatalf("expected context.Canceled, got %v", err)
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatal("cancelled waiter did not return")
	}

	deadline := time.After(100 * time.Millisecond)
	for q.Len() != 2 {
		select {
		case <-deadline:
			t.Fatalf("expected 2 waiters after cancel, got %d", q.Len())
		default:
			time.Sleep(time.Microsecond)
		}
	}
	verifyInvariants(t, q)

	// The surviving waiters must still be woken by the same txnID.
	q.NotifyUpTo(100)
	for i := 0; i < 2; i++ {
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("expected nil error for surviving waiter, got %v", err)
			}
		case <-time.After(100 * time.Millisecond):
			t.Fatal("surviving waiter not notified")
		}
	}

	cancel1() // no-op cleanup path: waiter was already removed by notify
	if q.Len() != 0 {
		t.Fatalf("expected empty queue, got %d", q.Len())
	}
}

func TestTxnWaitQueue_RandomCancellationsKeepInvariants(t *testing.T) {
	q := NewTxnWaitQueue()

	const n = 300
	type waiter struct {
		ctx    context.Context
		cancel context.CancelFunc
		done   chan error
	}
	waiters := make([]waiter, n)
	for i := range waiters {
		ctx, cancel := context.WithCancel(context.Background())
		waiters[i] = waiter{
			ctx:    ctx,
			cancel: cancel,
			done:   make(chan error, 1),
		}
		go func(w waiter, txnID uint64) {
			w.done <- q.Wait(w.ctx, txnID)
		}(waiters[i], uint64(rand.Uint32()))
	}
	for q.Len() != n {
		time.Sleep(time.Microsecond)
	}

	// Cancel half in random order; the index map must keep every removal exact.
	order := rand.Perm(n)
	for _, idx := range order[:n/2] {
		waiters[idx].cancel()
		select {
		case err := <-waiters[idx].done:
			if err != context.Canceled {
				t.Fatalf("waiter %d: expected context.Canceled, got %v", idx, err)
			}
		case <-time.After(time.Second):
			t.Fatalf("waiter %d did not return after cancel", idx)
		}
		verifyInvariants(t, q)
	}

	if q.Len() != n/2 {
		t.Fatalf("expected %d waiters after cancellations, got %d", n/2, q.Len())
	}

	q.NotifyUpTo(^uint64(0))
	for _, idx := range order[n/2:] {
		select {
		case err := <-waiters[idx].done:
			if err != nil {
				t.Fatalf("surviving waiter %d got error: %v", idx, err)
			}
		case <-time.After(time.Second):
			t.Fatalf("surviving waiter %d not notified", idx)
		}
	}

	for _, w := range waiters {
		w.cancel()
	}
	if q.Len() != 0 {
		t.Fatalf("expected empty queue, got %d", q.Len())
	}
	verifyInvariants(t, q)
}

func TestTxnWaitQueue_SingleWaiter(t *testing.T) {
	q := NewTxnWaitQueue()

	done := make(chan error, 1)
	go func() {
		done <- q.Wait(context.Background(), 100)
	}()

	// Give waiter time to register
	time.Sleep(10 * time.Millisecond)

	if q.Len() != 1 {
		t.Fatalf("expected 1 waiter, got %d", q.Len())
	}

	q.NotifyUpTo(100)

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("expected nil error, got %v", err)
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatal("waiter not notified")
	}

	if q.Len() != 0 {
		t.Fatalf("expected 0 waiters after notification, got %d", q.Len())
	}
}

func TestTxnWaitQueue_MultipleWaiters(t *testing.T) {
	q := NewTxnWaitQueue()

	var wg sync.WaitGroup
	results := make([]error, 5)

	txnIDs := []uint64{10, 20, 30, 40, 50}
	for i, txnID := range txnIDs {
		wg.Add(1)
		go func(idx int, id uint64) {
			defer wg.Done()
			results[idx] = q.Wait(context.Background(), id)
		}(i, txnID)
	}

	// Give waiters time to register
	time.Sleep(20 * time.Millisecond)

	if q.Len() != 5 {
		t.Fatalf("expected 5 waiters, got %d", q.Len())
	}

	// Notify up to 30 - should wake first 3 waiters
	q.NotifyUpTo(30)

	// Give some time for waiters to wake up
	time.Sleep(20 * time.Millisecond)

	if q.Len() != 2 {
		t.Fatalf("expected 2 waiters remaining, got %d", q.Len())
	}

	// Notify remaining
	q.NotifyUpTo(50)

	wg.Wait()

	for i, err := range results {
		if err != nil {
			t.Errorf("waiter %d got error: %v", i, err)
		}
	}

	if q.Len() != 0 {
		t.Fatalf("expected 0 waiters after all notifications, got %d", q.Len())
	}
}

func TestTxnWaitQueue_ContextCancellation(t *testing.T) {
	q := NewTxnWaitQueue()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)

	go func() {
		done <- q.Wait(ctx, 100)
	}()

	// Give waiter time to register
	time.Sleep(10 * time.Millisecond)

	if q.Len() != 1 {
		t.Fatalf("expected 1 waiter, got %d", q.Len())
	}

	cancel()

	select {
	case err := <-done:
		if err != context.Canceled {
			t.Fatalf("expected context.Canceled, got %v", err)
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatal("waiter did not return on cancel")
	}

	// Give cleanup time to run
	time.Sleep(10 * time.Millisecond)

	if q.Len() != 0 {
		t.Fatalf("expected 0 waiters after cancel, got %d", q.Len())
	}
}

func TestTxnWaitQueue_NotifyBeforeWait(t *testing.T) {
	q := NewTxnWaitQueue()

	// Notify before anyone waits
	q.NotifyUpTo(50)

	// Now wait for a txnID that's already passed
	done := make(chan error, 1)
	go func() {
		done <- q.Wait(context.Background(), 100)
	}()

	// This should still wait since queue doesn't track history
	time.Sleep(20 * time.Millisecond)

	if q.Len() != 1 {
		t.Fatalf("expected 1 waiter, got %d", q.Len())
	}

	q.NotifyUpTo(100)

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("expected nil error, got %v", err)
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatal("waiter not notified")
	}
}

func TestTxnWaitQueue_DuplicateTxnIDs(t *testing.T) {
	q := NewTxnWaitQueue()

	var wg sync.WaitGroup
	results := make([]error, 3)

	// Multiple waiters for same txnID
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx] = q.Wait(context.Background(), 100)
		}(i)
	}

	// Give waiters time to register
	time.Sleep(20 * time.Millisecond)

	if q.Len() != 3 {
		t.Fatalf("expected 3 waiters, got %d", q.Len())
	}

	// Single notify should wake all
	q.NotifyUpTo(100)

	wg.Wait()

	for i, err := range results {
		if err != nil {
			t.Errorf("waiter %d got error: %v", i, err)
		}
	}

	if q.Len() != 0 {
		t.Fatalf("expected 0 waiters after notification, got %d", q.Len())
	}
}

func TestTxnWaitQueue_PartialNotify(t *testing.T) {
	q := NewTxnWaitQueue()

	var wg sync.WaitGroup

	// Start waiters for txnIDs: 10, 20, 30
	txnIDs := []uint64{10, 20, 30}
	notified := make([]atomic.Bool, 3)

	for i, txnID := range txnIDs {
		wg.Add(1)
		go func(idx int, id uint64) {
			defer wg.Done()
			if err := q.Wait(context.Background(), id); err == nil {
				notified[idx].Store(true)
			}
		}(i, txnID)
	}

	// Give waiters time to register
	time.Sleep(20 * time.Millisecond)

	// Notify up to 15 - should only wake waiter for txnID 10
	q.NotifyUpTo(15)

	// Give some time for notification
	time.Sleep(20 * time.Millisecond)

	if !notified[0].Load() {
		t.Error("waiter for txnID 10 should be notified")
	}
	if notified[1].Load() || notified[2].Load() {
		t.Error("waiters for txnID 20, 30 should not be notified yet")
	}

	if q.Len() != 2 {
		t.Fatalf("expected 2 waiters remaining, got %d", q.Len())
	}

	// Notify rest
	q.NotifyUpTo(30)
	wg.Wait()

	if !notified[1].Load() || !notified[2].Load() {
		t.Error("all waiters should be notified")
	}
}

func TestTxnWaitQueue_Timeout(t *testing.T) {
	q := NewTxnWaitQueue()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	start := time.Now()
	err := q.Wait(ctx, 100)
	duration := time.Since(start)

	if err != context.DeadlineExceeded {
		t.Fatalf("expected context.DeadlineExceeded, got %v", err)
	}

	if duration < 40*time.Millisecond || duration > 100*time.Millisecond {
		t.Fatalf("timeout took %v, expected ~50ms", duration)
	}

	// Give cleanup time to run
	time.Sleep(10 * time.Millisecond)

	if q.Len() != 0 {
		t.Fatalf("expected 0 waiters after timeout, got %d", q.Len())
	}
}

func TestTxnWaitQueue_ConcurrentWaitAndNotify(t *testing.T) {
	q := NewTxnWaitQueue()

	var wg sync.WaitGroup
	var successCount atomic.Int32

	// Start 100 waiters
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(txnID uint64) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
			defer cancel()
			if err := q.Wait(ctx, txnID); err == nil {
				successCount.Add(1)
			}
		}(uint64(i))
	}

	// Give waiters time to register
	time.Sleep(50 * time.Millisecond)

	// Concurrently notify in batches
	for i := 0; i < 10; i++ {
		q.NotifyUpTo(uint64((i + 1) * 10))
		time.Sleep(10 * time.Millisecond)
	}

	wg.Wait()

	if successCount.Load() != 100 {
		t.Errorf("expected 100 successful waits, got %d", successCount.Load())
	}

	if q.Len() != 0 {
		t.Fatalf("expected 0 waiters remaining, got %d", q.Len())
	}
}

func BenchmarkTxnWaitQueue_SingleWaiter(b *testing.B) {
	q := NewTxnWaitQueue()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		go func(txnID uint64) {
			q.Wait(context.Background(), txnID)
		}(uint64(i))

		q.NotifyUpTo(uint64(i))
	}
}

func BenchmarkTxnWaitQueue_1000Waiters(b *testing.B) {
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		q := NewTxnWaitQueue()
		var wg sync.WaitGroup

		// Start 1000 waiters
		for j := 0; j < 1000; j++ {
			wg.Add(1)
			go func(txnID uint64) {
				defer wg.Done()
				q.Wait(context.Background(), txnID)
			}(uint64(j))
		}

		// Give goroutines time to register
		for q.Len() < 1000 {
			time.Sleep(time.Microsecond)
		}

		b.StartTimer()
		// Notify all at once
		q.NotifyUpTo(1000)
		wg.Wait()
		b.StopTimer()
	}
}

func BenchmarkTxnWaitQueue_BatchNotify(b *testing.B) {
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		q := NewTxnWaitQueue()
		var wg sync.WaitGroup

		// Start 1000 waiters
		for j := 0; j < 1000; j++ {
			wg.Add(1)
			go func(txnID uint64) {
				defer wg.Done()
				q.Wait(context.Background(), txnID)
			}(uint64(j))
		}

		// Give goroutines time to register
		for q.Len() < 1000 {
			time.Sleep(time.Microsecond)
		}

		b.StartTimer()
		// Notify in batches of 100
		for j := 0; j < 10; j++ {
			q.NotifyUpTo(uint64((j + 1) * 100))
		}
		wg.Wait()
		b.StopTimer()
	}
}

// BenchmarkTxnWaitQueue_CancelDense models the timeout-heavy workload:
// a deep queue where cancellations outnumber commits. Each iteration cancels
// 90% of a fresh 1024-entry queue and notifies the surviving 10%, measuring
// cancel-path cost without the O(n) linear scan.
func BenchmarkTxnWaitQueue_CancelDense(b *testing.B) {
	const total = 1024
	const cancelled = total - total/10

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		q := NewTxnWaitQueue()
		ctxs := make([]context.Context, total)
		cancels := make([]context.CancelFunc, total)
		done := make([]chan error, total)
		for j := range ctxs {
			ctxs[j], cancels[j] = context.WithCancel(context.Background())
			done[j] = make(chan error, 1)
			go func(ch chan error, ctx context.Context, txnID uint64) {
				ch <- q.Wait(ctx, txnID)
			}(done[j], ctxs[j], uint64(j))
		}
		for q.Len() != total {
			time.Sleep(time.Microsecond)
		}
		b.StartTimer()

		for j := 0; j < cancelled; j++ {
			cancels[j]()
		}
		for j := 0; j < cancelled; j++ {
			<-done[j]
		}

		// The remaining waiters are the highest txnIDs; commit them.
		q.NotifyUpTo(total)
		for j := cancelled; j < total; j++ {
			<-done[j]
		}
		b.StopTimer()

		for j := range cancels {
			cancels[j]()
		}
	}
}

// BenchmarkTxnWaitQueue_Insert measures steady-state insertion cost into a
// queue kept at a bounded depth, guarding against regressions from the index.
func BenchmarkTxnWaitQueue_Insert(b *testing.B) {
	const depth = 1024
	q := NewTxnWaitQueue()
	ctxs := make([]context.Context, depth)
	cancels := make([]context.CancelFunc, depth)
	for j := range ctxs {
		ctxs[j], cancels[j] = context.WithCancel(context.Background())
	}
	defer func() {
		for _, cancel := range cancels {
			cancel()
		}
	}()

	done := make(chan error, depth)
	for j := 0; j < depth; j++ {
		go func(ch chan error, ctx context.Context, txnID uint64) {
			ch <- q.Wait(ctx, txnID)
		}(done, ctxs[j], uint64(j))
	}
	for q.Len() != depth {
		time.Sleep(time.Microsecond)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx, cancel := context.WithCancel(context.Background())
		cancels = append(cancels, cancel)
		go q.Wait(ctx, uint64(depth+i))
		for q.Len() != depth+1 {
			time.Sleep(time.Microsecond)
		}
		cancel()
		for q.Len() != depth {
			time.Sleep(time.Microsecond)
		}
	}
}
