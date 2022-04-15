import numpy as np
import random


class PER:
    def __init__(self, state_space, action_space, max_size=100000, eps=1e-2, alpha=0.1, beta=0.1):
        self.tree = SumTree(max_size)

        # transition
        self.current_states = np.empty((0, state_space), dtype=np.float64)
        self.actions = np.empty((0, action_space), dtype=np.float64)
        self.rewards = np.empty((0, 1), dtype=np.float64)
        self.next_states = np.empty((0, state_space), dtype=np.float64)

        # PER parameters
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.max_priority = eps

        self.count = 0
        self.real_size = 0
        self.size = max_size

    def store(self, current_state, action, reward, next_state):
        self.tree.add(self.max_priority, self.count)

        self.current_states = np.append(self.current_states[-self.size:],
                                        np.array(current_state, ndmin=2), axis=0)
        self.actions = np.append(self.actions[-self.size:],
                                 np.array(action, ndmin=2), axis=0)
        self.rewards = np.append(self.rewards[-self.size:],
                                 np.array(reward, ndmin=2), axis=0)
        self.next_states = np.append(self.next_states[-self.size:],
                                     np.array(next_state, ndmin=2), axis=0)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def fetch_sample(self, num_samples):
        # assert self.real_size >= num_samples, "buffer contains less samples than batch size"
        if num_samples > self.real_size:
            num_samples = self.real_size

        sample_idxs, tree_idxs = [], []
        priorities = np.empty((num_samples, 1), dtype=np.float64)

        segment = self.tree.total() / num_samples

        for i in range(num_samples):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)

            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total

        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()

        current_state_ = self.current_states[sample_idxs]
        actions_ = self.actions[sample_idxs]
        rewards_ = self.rewards[sample_idxs]
        next_states_ = self.next_states[sample_idxs]

        self.update_priorities(sample_idxs, priorities)

        return current_state_, actions_, rewards_, next_states_, weights

    def update_priorities(self, data_idxs, priorities):
        for data_idx, priority in zip(data_idxs, priorities):
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1

        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[dataIdx]
