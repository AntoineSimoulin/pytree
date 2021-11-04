
def prepare_input_from_dependency_tree(dependency_tree):
    head_idx = [int(t.strip().split('\t')[6]) for t in dependency_tree.strip().split('\n')]
    tokens = [t.strip().split('\t')[1] for t in dependency_tree.strip().split('\n')]
    return tokens, head_idx
