import re


def prepare_input_from_constituency_tree(constituency_tree):
    cons_tree = ConsTree([])
    tree = cons_tree.read_tree(constituency_tree[5:-1])
    tree.close_unaries()
    tree.left_markovize(dummy_annotation="")
    const = cons_tree.linearize_parse_tree(str(tree))
    clean_const = re.sub(r'\(([^ ]+) ', r'([\1] ', const)

    tokens = []
    for token in re.sub(r'\(([^ ]+) ', r'([\1] ', clean_const).split():
        token = token.strip('()')
        tokens.append(token)
    n_original_tokens = len([t for t in tokens if not t.startswith('[') and not t.endswith(']')])
    n_tokens = len(tokens)
    
    clean_const = re.sub(r'\)', r' )', clean_const)

    n_original_tokens_idx = 0
    n_specific_tokens_idx = n_original_tokens
    clean_const_idx = ""
    vocab = [''] * n_tokens
    
    for t in clean_const.split():
        if t != ')':
            if not t[1:].startswith('[') and not t.endswith(']'):
                clean_const_idx += str(n_original_tokens_idx)
                vocab[n_original_tokens_idx] = t
                n_original_tokens_idx += 1
            else:
                clean_const_idx += '('
                clean_const_idx += str(n_specific_tokens_idx)
                vocab[n_specific_tokens_idx] = t[1:]
                n_specific_tokens_idx += 1
        else:
            clean_const_idx += ')'
        clean_const_idx += ' '
    
    clean_const_idx = ''.join(clean_const_idx[:-1])

    head_idx_ = [0] * n_tokens

    regexp = re.compile(r'\((\d+) (\d+) (\d+) \)')
    while regexp.search(clean_const_idx):
        for (head_idx, child_1_idx, child_2_idx) in re.findall(regexp, clean_const_idx):
            head_idx_[int(child_1_idx)] = int(head_idx)
            head_idx_[int(child_2_idx)] = int(head_idx)
        clean_const_idx = re.sub(r'\((\d+) \d+ \d+ \)', r'\1', clean_const_idx)

    return vocab, head_idx_


class ConsTree(object):
    """
    That's your phrase structure tree.
    """

    def __init__(self, label, children=None):
        self.label = label
        self.children = [] if children is None else children

    def copy(self):
        """
        Performs a deep copy of this tree
        """
        return ConsTree(self.label, [c.copy() for c in self.children])

    @staticmethod
    def linearize_parse_tree(parse_tree):
        linearized_tree = re.sub(r'(:? ?\([^(]*\_ )([^)]*)\)', r' \2', parse_tree)
        # linearized_tree = [x.rstrip() for x in re.split('([\(\)])', parse_tree) if x.rstrip()]
        # linearized_tree = [x for x in linearized_tree[2:] if x != "("]
        return linearized_tree

    def is_leaf(self):
        return self.children == []

    def add_child(self, child_node):
        self.children.append(child_node)

    def arity(self):
        return len(self.children)

    def get_child(self, idx=0):
        """
        @return the idx-th child of this node.
        """
        return self.children[idx]

    def __str__(self):
        """
        Pretty prints the tree
        """
        return self.label if self.is_leaf() else '(%s %s)' % (
        self.label, ' '.join([str(child) for child in self.children]))

    def tokens(self, labels=True):
        """
        @param labels: returns a list of strings if true else returns
        a list of ConsTree objects (leaves)
        @return the list of words at the leaves of the tree
        """
        if self.is_leaf():
            return [self.label] if labels else [self]
        else:
            result = []
            for child in self.children:
                result.extend(child.tokens(labels))
            return result

    def pos_tags(self):
        """
        @return the list of pos tags as ConsTree objects
        """
        if self.arity() == 1 and self.get_child().is_leaf():
            return [self]
        else:
            result = []
            for child in self.children:
                result.extend(child.pos_tags())
            return result

    def index_leaves(self):
        """
        Adds an numeric index to each leaf node
        """
        for idx, elt in enumerate(self.tokens(labels=False)):
            elt.idx = idx

    def triples(self):
        """
        Extracts a list of evalb triples from the tree
        (supposes leaves are indexed)
        """
        subtriples = []
        if self.is_leaf():
            return [(self.idx, self.idx + 1, self.label)]

        for child in self.children:
            subtriples.extend(child.triples())
        leftidx = min([idx for idx, jdx, label in subtriples])
        rightidx = max([jdx for idx, jdx, label in subtriples])
        subtriples.append((leftidx, rightidx, self.label))
        return subtriples

    def compare(self, other):
        """
        Compares this tree to another and computes precision,recall,
        fscore. Assumes self is the reference tree
        @param other: the predicted tree
        @return (precision,recall,fscore)
        """
        print('***', str(self), str(other))

        self.index_leaves()
        other.index_leaves()

        # filter out leaves
        # ref_triples  = set([(i,j,X) for i,j,X in self.triples() if j != i+1])
        # pred_triples = set([(i,j,X) for i,j,X in other.triples() if j != i+1])

        ref_triples = set(self.triples())
        pred_triples = set(other.triples())

        intersect = ref_triples.intersection(pred_triples)
        isize = len(intersect)
        P = isize / len(pred_triples)
        R = isize / len(ref_triples)
        F = (2 * P * R) / (P + R)
        return (P, R, F)

    def strip_tags(self):
        """
        In place (destructive) removal of pos tags
        """

        def gen_child(node):
            if len(node.children) == 1 and node.children[0].is_leaf():
                return node.children[0]
            return node

        self.children = [gen_child(child) for child in self.children]
        for child in self.children:
            child.strip_tags()

    def normalize_OOV(self, lexicon, unk_token):
        """
        Destructively replaces all leaves by the unk_token when the leaf label is not in
        lexicon. Normalizes numbers
        @param lexicon  : a set of strings
        @param unk_token: a string
        @return a pointer to the tree root
        """
        if self.is_leaf():
            if self.label not in lexicon:
                self.label = unk_token
        for child in self.children:
            child.normalize_OOV(lexicon, unk_token)
        return self

    def add_gold_tags(self, tag_sequence=None, idx=0):
        """
        Adds gold tags to the tree on top of leaves(for evalb compatibility).
        Destructive method.
        """
        newchildren = []
        for child in self.children:
            if child.is_leaf():
                label = tag_sequence[idx]
                tag = ConsTree(label, children=[child])
                newchildren.append(tag)
                idx += 1
            else:
                newchildren.append(child)
                idx = child.add_gold_tags(tag_sequence, idx)
        self.children = newchildren
        return idx

    def add_dummy_root(self, root_label='TOP'):
        """
        In place addition of a dummy root
        """
        selfcopy = ConsTree(self.label, children=self.children)
        self.children = [selfcopy]
        self.label = root_label

    def close_unaries(self, dummy_annotation='@'):
        """
        In place (destructive) unary closure of unary branches
        """
        if self.arity() == 1:
            current = self
            unary_labels = []
            while current.arity() == 1 and not current.get_child().is_leaf():
                unary_labels.append(current.label)
                current = current.get_child()
            unary_labels.append(current.label)
            self.label = dummy_annotation.join(unary_labels)
            self.children = current.children

        for child in self.children:
            child.close_unaries()

    def expand_unaries(self, dummy_annotation='@'):
        """
        In place (destructive) expansion of unary symbols.
        """
        if dummy_annotation in self.label:
            unary_chain = self.label.split(dummy_annotation)
            self.label = unary_chain[0]
            backup = self.children
            current = self
            for label in unary_chain[1:]:
                c = ConsTree(label)
                current.children = [c]
                current = c
            current.children = backup

        for child in self.children:
            child.expand_unaries()

    def left_markovize(self, dummy_annotation=':'):
        """
        In place (destructive) left markovization (order 0)
        """
        if len(self.children) > 2:
            left_sequence = self.children[:-1]
            dummy_label = self.label if self.label[-1] == dummy_annotation else self.label + dummy_annotation
            dummy_tree = ConsTree(dummy_label, left_sequence)
            self.children = [dummy_tree, self.children[-1]]
        for child in self.children:
            child.left_markovize()

    def right_markovize(self, dummy_annotation=':'):
        """
        In place (destructive) right markovization (order 0)
        """
        if len(self.children) > 2:
            right_sequence = self.children[1:]
            dummy_label = self.label if self.label[-1] == dummy_annotation else self.label + dummy_annotation
            dummy_tree = ConsTree(dummy_label, right_sequence)
            self.children = [self.children[0], dummy_tree]
        for child in self.children:
            child.right_markovize()

    def unbinarize(self, dummy_annotation=':'):
        """
        In place (destructive) unbinarization
        """
        newchildren = []
        for child in self.children:
            if child.label[-1] == dummy_annotation:
                child.unbinarize()
                newchildren.extend(child.children)
            else:
                child.unbinarize()
                newchildren.append(child)
        self.children = newchildren

    def collect_nonterminals(self):
        """
        Returns the list of nonterminals found in a tree:
        """
        if not self.is_leaf():
            result = [self.label]
            for child in self.children:
                result.extend(child.collect_nonterminals())
            return result
        return []

    @staticmethod
    def read_tree(input_str):
        """
        Reads a one line s-expression.
        This is a non robust function to syntax errors
        @param input_str: a s-expr string
        @return a ConsTree object
        """
        tokens = input_str.replace('(', ' ( ').replace(')', ' ) ').split()
        stack = [ConsTree('dummy')]
        for idx, tok in enumerate(tokens):
            if tok == '(':
                current = ConsTree(tokens[idx + 1])
                stack[-1].add_child(current)
                stack.append(current)
            elif tok == ')':
                stack.pop()
            else:
                if tokens[idx - 1] != '(':
                    stack[-1].add_child(ConsTree(tok))
        assert (len(stack) == 1)
        return stack[-1].get_child()
