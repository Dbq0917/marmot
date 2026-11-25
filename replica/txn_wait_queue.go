package replica

import (
	"context"
	"sort"
	"sync"
)

type txnWaiter struct {
	txnID uint64
	ch    chan struct{}
}

// TxnWaitQueue manages waiters for transaction replication.
// Uses a sorted list and a monotonic watermark so each notification only
// inspects the waiters it satisfies.
type TxnWaitQueue struct {
	mu       sync.Mutex
	waiters  []txnWaiter // Sorted by txnID ascending
	notified uint64      // Highest txnID observed by NotifyUpTo
}

func NewTxnWaitQueue() *TxnWaitQueue {
	return &TxnWaitQueue{
		waiters: make([]txnWaiter, 0),
	}
}

// Wait blocks until txnID is replicated or context is cancelled.
// Returns nil if txnID reached, context error otherwise.
func (q *TxnWaitQueue) Wait(ctx context.Context, txnID uint64) error {
	ch := make(chan struct{})

	q.mu.Lock()
	if txnID <= q.notified {
		q.mu.Unlock()
		return nil
	}

	// Insert in sorted order (binary search for position)
	i := sort.Search(len(q.waiters), func(i int) bool {
		return q.waiters[i].txnID >= txnID
	})
	// Insert at position i
	q.waiters = append(q.waiters, txnWaiter{})
	copy(q.waiters[i+1:], q.waiters[i:])
	q.waiters[i] = txnWaiter{txnID: txnID, ch: ch}
	q.mu.Unlock()

	select {
	case <-ch:
		return nil
	case <-ctx.Done():
		// Remove ourselves from queue on cancel
		q.mu.Lock()
		for j, w := range q.waiters {
			if w.ch == ch {
				q.waiters = append(q.waiters[:j], q.waiters[j+1:]...)
				break
			}
		}
		q.mu.Unlock()
		return ctx.Err()
	}
}

// NotifyUpTo advances the committed watermark and wakes all waiters with
// txnID <= the watermark. Transaction IDs may have gaps, and a notification
// that does not advance the watermark is ignored.
// Called when a transaction is applied from the replication stream.
// O(log n) search + O(k) notifications where k = satisfied waiters.
func (q *TxnWaitQueue) NotifyUpTo(txnID uint64) {
	q.mu.Lock()
	defer q.mu.Unlock()

	if txnID <= q.notified {
		return
	}
	q.notified = txnID

	// Binary search for first waiter beyond the new watermark.
	i := sort.Search(len(q.waiters), func(i int) bool {
		return q.waiters[i].txnID > q.notified
	})

	// Close channels for all waiters with txnID <= watermark.
	for j := 0; j < i; j++ {
		close(q.waiters[j].ch)
	}

	// Remove satisfied waiters
	q.waiters = q.waiters[i:]
}

// Len returns number of active waiters (for testing/metrics)
func (q *TxnWaitQueue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.waiters)
}
