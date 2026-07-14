import javalang
parse_errors = 0

class TreeNode:
    __slots__ = ["token", "children"]
    def __init__(self, token, children=None):
        self.token    = token
        self.children = children or []
    def to_dict(self):
        return {"token": self.token,
                "children": [c.to_dict() for c in self.children]}

def build_tree(node, max_depth=60, _d=0):
    if not isinstance(node, javalang.ast.Node):
        return None
    if _d > max_depth:
        return TreeNode("<DEEP>")

    token    = node.__class__.__name__
    children = []
    for attr in node.children:
        if isinstance(attr, (list, tuple)):
            for item in attr:
                if isinstance(item, javalang.ast.Node):
                    c = build_tree(item, max_depth, _d + 1)
                    if c:
                        children.append(c)
        elif isinstance(attr, javalang.ast.Node):
            c = build_tree(attr, max_depth, _d + 1)
            if c:
                children.append(c)
    return TreeNode(token, children)

def java_file_to_ast(file_path: str):
    global parse_errors
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        if len(code) > 300_000:          
            return None
        tree = javalang.parse.parse(code)
        return build_tree(tree)
    except Exception:
        parse_errors += 1
        return None

def java_code_to_ast(code_str: str):
    try:
        tree = javalang.parse.parse(code_str)
        return build_tree(tree), None
    except Exception as e:
        return None, str(e)