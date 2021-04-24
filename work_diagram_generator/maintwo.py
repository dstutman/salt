from dataclasses import dataclass
from datetime import timedelta

@dataclass
class Task:
    name: str
    desc: str = ''
    assignee: str = ''
    ident: str = None
    duration: timedelta = None
    parent: 'Task' = None
    children: list['Task'] = None
    previous: list['Task'] = None
    then: list['Task'] = None

def patch_relationships(task: Task):
    if task.children is None:
        task.children = []
        
    if task.then is None:
        task.then= []

    if task.previous is not None:
        for prev in task.previous:
            if prev.then is None:
                prev.then = []

            if not task in prev.then:
                prev.then.append(task)
            patch_relationships(prev)

    if task.parent is not None:
        if task.parent.children is None:
            task.parent.children = []
        
        if not task in task.parent.children:
            task.parent.children.append(task)
        patch_relationships(task.parent)

def assign_sub_ids(task: Task, root, set_ident=None):
    ident = []
    if set_ident is None:
        ident = ['1', 'a']
    else:
        ident = set_ident.copy()

    task.ident = ''.join(ident)

    if task.then is None:
        return

    tmp_ident = ident.copy()
    for nxt in task.then:
        if nxt in task.children:
            assign_sub_ids(nxt, root, set_ident=tmp_ident + ['.', '1', 'a'])
        #tmp_ident[-1] = chr(ord(tmp_ident[-1])+1)

    tmp_ident = ident.copy()
    tmp_ident[-2] = chr(ord(tmp_ident[-2])+1)
    tmp_ident[-1] = 'a'
    for nxt in task.then:
        if nxt not in task.children and nxt.parent == task.parent:
            assign_sub_ids(nxt, root, set_ident=tmp_ident)
            tmp_ident[-1] = chr(ord(tmp_ident[-1])+1)

    for nxt in task.then:
        if nxt not in task.children and nxt.parent != task.parent:
            tmp_ident = ident.copy()
            tmp_ident = tmp_ident[0:-3]
            tmp_ident[-2] = chr(ord(tmp_ident[-2])+1)
            assign_sub_ids(nxt, root, set_ident=tmp_ident)

def slug(input: str) -> str:
    return input.replace(" ", "_")

def gen_flow(task: Task, is_root=True) -> str:
    mermaid = ''
    if is_root:
        mermaid = 'graph LR\n'
    
    mermaid += f'{slug(task.name)}[{task.name}<br>{task.ident}]\n'
    for nxt in task.then:
        mermaid += f'{slug(task.name)} --> {slug(nxt.name)}\n'
        mermaid += gen_mermaid(nxt, is_root=False)

    return mermaid

def gen_gantt(task: Task, is_root=True) -> str:
    mermaid = ''
    if is_root:
        mermaid = 'graph LR\n'
    
    mermaid += f'{slug(task.name)}[{task.name}<br>{task.ident}]\n'
    for nxt in task.then:
        mermaid += f'{slug(task.name)} --> {slug(nxt.name)}\n'
        mermaid += gen_mermaid(nxt, is_root=False)

    return mermaid

if __name__ == '__main__':
    root = Task('Root', assignee='p1')
    root_child = Task('Root child', parent=root, previous=[root], assignee='p1')
    root_child_then = Task('Root child then', parent=root, previous=[root_child], assignee='p1')
    root_child_thentwo = Task('Root child then two', parent=root, previous=[root_child], assignee='p1')
    root_child_thenthree = Task('Root child then three', parent=root, previous=[root_child], assignee='p1')
    join = Task('Join', previous=[root_child_then, root_child_thentwo, root_child_thenthree], assignee='p1')
    jointwo = Task('Join two', previous=[root_child_then, root_child_thentwo, root_child_thenthree], assignee='p1')
    ext2 = Task('Exit2', previous=[join, jointwo], assignee='p1')
    patch_relationships(ext2)
    assign_sub_ids(root, root)
    print(gen_mermaid(root))