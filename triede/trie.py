from dataclasses import dataclass, field
import heapq

@dataclass
class TrieNode:
    token: object
    child: dict = field(default_factory=dict) 
    score: int = 0

    def get_child(self, token):
        if token not in self.child:
            self.child[token] = TrieNode(token=token, score=0)

        return self.child[token]

    def get_child_seq(self, seq_token):
        if not seq_token:
            return self

        node = self.get_child(seq_token[0])
        return node.get_child_seq(seq_token[1:])

    def yield_child_seq(self, seq_len):
        if not self.child or not seq_len:
            yield self.score, []

        for token, child in self.child.items():
            for score, child_seq in child.yield_child_seq(seq_len-1):
                yield score, [token, ] + child_seq

    def inc_score(self):
        self.score += 1

    def record(self, seq_token):
        if not seq_token:
            return

        node = self.get_child(seq_token[0])
        node.inc_score()
        node.record(seq_token[1:])



class Trie(TrieNode):
    def gen_by_prefix(self, prefix, guess_size, topk=0):
        node = self.get_child_seq(prefix)
        queue = []
        for score, child_seq in node.yield_child_seq(guess_size):
            if len(child_seq) != guess_size:
                continue
            heapq.heappush(queue, (score, child_seq))

            if topk > 0 and len(queue) > topk:
                heapq.heappop(queue)


        return [child_seq for _, child_seq in queue]
