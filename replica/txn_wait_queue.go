package replica

import (
	"context"
	"sync"
)

type txnWaiter struct {
	txnID uint64
	ch    chan struct{}
}

// TxnWaitQueue manages waiters for transaction replication.
//
// Waiters are kept in a binary min-heap ordered by txnID, with a side index
// mapping each waiter's channel to its current heap position. Insertion and
// cancellation are both O(log n): the index turns cancel-time lookup from a
// full O(n) scan into O(1) position lookup followed by O(log n) re-heap.
// Notification pops satisfied waiters off the heap top in O(k log n) for
// k satisfied waiters.
type TxnWaitQueue struct {
	mu      sync.Mutex
	waiters []txnWaiter           // min-heap ordered by txnID
	index   map[chan struct{}]int // waiter channel -> position in waiters
}

func NewTxnWaitQueue() *TxnWaitQueue {
	return &TxnWaitQueue{
		waiters: make([]txnWaiter, 0),
		index:   make(map[chan struct{}]int),
	}
}

func (q *TxnWaitQueue) less(i, j int) bool {
	return q.waiters[i].txnID < q.waiters[j].txnID
}

// swap updates the heap and the position index for both moved waiters.
// Caller must hold q.mu.
func (q *TxnWaitQueue) swap(i, j int) {
	q.waiters[i], q.waiters[j] = q.waiters[j], q.waiters[i]
	q.index[q.waiters[i].ch] = i
	q.index[q.waiters[j].ch] = j
}

// siftUp restores the heap property toward the root. Caller must hold q.mu.
func (q *TxnWaitQueue) siftUp(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if !q.less(i, parent) {
			return
		}
		q.swap(i, parent)
		i = parent
	}
}

// siftDown restores the heap property toward the leaves. Caller must hold q.mu.
func (q *TxnWaitQueue) siftDown(i, n int) {
	for {
		left := 2*i + 1
		if left >= n {
			return
		}
		smallest := left
		if right := left + 1; right < n && q.less(right, left) {
			smallest = right
		}
		if !q.less(smallest, i) {
			return
		}
		q.swap(i, smallest)
		i = smallest
	}
}

// push appends a waiter and re-heaps. Caller must hold q.mu.
func (q *TxnWaitQueue) push(w txnWaiter) {
	i := len(q.waiters)
	q.waiters = append(q.waiters, w)
	q.index[w.ch] = i
	q.siftUp(i)
}

// removeAt removes the waiter at heap position i and re-heaps.
// Caller must hold q.mu.
func (q *TxnWaitQueue) removeAt(i int) txnWaiter {
	removed := q.waiters[i]
	delete(q.index, removed.ch)

	n := len(q.waiters) - 1
	if i == n {
		q.waiters = q.waiters[:n]
		return removed
	}

	// Move the last element into the vacated slot, then fix whichever
	// direction violates the heap property.
	q.waiters[i] = q.waiters[n]
	q.waiters = q.waiters[:n]
	q.index[q.waiters[i].ch] = i
	if i > 0 && q.less(i, (i-1)/2) {
		q.siftUp(i)
	} else {
		q.siftDown(i, n)
	}
	return removed
}

// Wait blocks until txnID is replicated or context is cancelled.
// Returns nil if txnID reached, context error otherwise.
func (q *TxnWaitQueue) Wait(ctx context.Context, txnID uint64) error {
	ch := make(chan struct{})

	q.mu.Lock()
	q.push(txnWaiter{txnID: txnID, ch: ch})
	q.mu.Unlock()

	select {
	case <-ch:
		return nil
	case <-ctx.Done():
		// Remove ourselves from queue on cancel. The index map locates our
		// heap position in O(1) instead of scanning every waiter.
		q.mu.Lock()
		if pos, ok := q.index[ch]; ok {
			q.removeAt(pos)
		}
		q.mu.Unlock()
		return ctx.Err()
	}
}

// NotifyUpTo wakes all waiters with txnID <= the given txnID.
// Called when a transaction is applied from the replication stream.
// O(k log n) where k = satisfied waiters, each popped off the heap top.
func (q *TxnWaitQueue) NotifyUpTo(txnID uint64) {
	q.mu.Lock()
	defer q.mu.Unlock()

	// Every satisfied waiter (txnID <= threshold) must sit at the heap top
	// while the minimum does, so pop until the top exceeds the threshold.
	for len(q.waiters) > 0 && q.waiters[0].txnID <= txnID {
		w := q.removeAt(0)
		close(w.ch)
	}
}

// Len returns number of active waiters (for testing/metrics)
func (q *TxnWaitQueue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.waiters)
}
