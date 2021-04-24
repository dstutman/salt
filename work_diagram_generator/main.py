from dataclasses import dataclass
from typing import Optional, Union, NamedTuple
from datetime import datetime, timedelta
from itertools import accumulate, combinations

@dataclass(frozen=True)
class Task:
    name: str
    desc: Optional[str] = None
    duration: timedelta = None
    assignee: str = None

@dataclass(frozen=True)
class ParentChild:
    parent: 'Task'
    child: 'Task'


@dataclass(frozen=True)
class BeforeAfter:
    before: 'Task'
    after: 'Task'

@dataclass(frozen=True)
class Sibling:
    first: 'Task'
    second: 'Task'
    def __hash__(self):
        # This is really really bad and will definitely break in an unpleasant way
        return len(self.first.name) + len(self.second.name)

@dataclass
class TaskWrapper:
    task: Task
    # Task's parent
    parent: Optional['TaskWrapper'] = None
    # Task's siblings
    #siblings: Optional[list['TaskWrapper']] = None
    # Task's antecedent
    previous: Optional[list['TaskWrapper']] = None
    # Tasks's children
    _children: Optional[list['TaskWrapper']] = None
    # Task's ancestors
    _next: Optional[list['TaskWrapper']] = None

def enumerate_most_tasks_and_rels(task_wrapper: TaskWrapper) -> (set[Task], set[any]):
    tasks = set()
    tasks.add(task_wrapper.task)

    rels = set()

    parent = traverse_to_parent(task_wrapper)
    if parent is not None:
        # We do not add the parent to tasks here because it must be
        # in the next -> prev relations as well eventually
        rels.add(ParentChild(parent.task, task_wrapper.task))

    #if task_wrapper.siblings is not None:
    #    tasks.update([sibl.task for sibl in task_wrapper.siblings])
    #    rels.update([Sibling(task_wrapper.task, sibl.task) for sibl in task_wrapper.siblings])

    if task_wrapper.previous is not None:
        tasks.update([prev.task for prev in task_wrapper.previous])
        rels.update([BeforeAfter(prev.task, task_wrapper.task) for prev in task_wrapper.previous])
        for predecessor in task_wrapper.previous:
            pretask, prerel = enumerate_most_tasks_and_rels(predecessor)
            tasks.update(pretask)
            rels.update(prerel)

    return (tasks, rels)

def patch_sibling_rels(tasks: set[Task], rels: set[any]) -> (set[Task], set[any]):
    pc_rels = set(filter_instanceof(ParentChild, rels))
    groupings = []
    for task in tasks:
        task_group = []
        for rel in pc_rels:
            if rel.parent == task:
                task_group.append(rel)
        groupings.append(task_group)

    for group in groupings:
        rels.update([Sibling(pair[0].child, pair[1].child) for pair in combinations(group, 2)])

    return (tasks, rels)

"""
Find a task's parent by traversing 'after' pointers
"""
def traverse_to_parent(task_wrapper: TaskWrapper) -> TaskWrapper:
    if task_wrapper.parent is not None:
        # Trivial case, this tasks parent is the parent of
        # the initial task
        return task_wrapper.parent
    # In a sense the below case would take an ill-formed tree
    # and convert it to an interpretation of the tree, so I won't
    # allow it for now, unless it proves exceptionally ergonomic
    #elif task.after is not None:
    #    # The parent is somewhere earlier in the dependencies
    #    # of this task
    #    return traverse_to_parent(task.after[0])
    else:
        # This case will only be true for the
        # root task (in a well formed tree)
        return None

def slug(input: str) -> str:
    return input.replace(" ", "_")

def filter_instanceof(ofwhat, set: set[any]) -> any:
    return filter(lambda elem: isinstance(elem, ofwhat), set)

def find_parent(task: Task, tasks: set[Task], rels: set[any]) -> Task:
    maybeparent = set(filter(lambda rel: rel.child == task, filter_instanceof(ParentChild, rels)))
    return next(iter(maybeparent)).parent if len(maybeparent) > 0 else None

def find_children(task: Task, tasks: set[Task], rels: set[any]) -> set[Task]:
    maybechildren = set(filter(lambda rel: rel.parent == task, filter_instanceof(ParentChild, rels)))
    return set(map(lambda pc: pc.child, maybechildren)) if len(maybechildren) > 0 else None

def find_siblings(task: Task, tasks: set[Task], rels: set[any]) -> set[Task]:
    maybesiblings = set(filter(lambda rel: rel.first == task or rel.second == task, filter_instanceof(Sibling, rels)))
    return set(map(lambda sb: sb.first if sb.second == task else sb.second, maybesiblings)) if len(maybesiblings) > 0 else None

def find_prev(task: Task, tasks: set[Task], rels: set[any]) -> Task:
    maybeprev = set(filter(lambda rel: rel.after == task, filter_instanceof(BeforeAfter, rels)))
    return next(iter(maybeprev)).before if len(maybeprev) > 0 else None

def get_seq_num(task: Task, tasks: set[Task], rels: set[any]) -> int:
    num = 0
    earlier = task
    while earlier != find_parent(task, tasks, rels):
        earlier = find_prev(earlier, tasks, rels)
        num += 1
    return num

def get_task_id(task: Task, tasks: set[Task], rels: set[any]) -> str:
    path = []
    curr = task
    while curr != None:
        path.append(str(get_seq_num(curr, tasks, rels)))
        siblings = find_siblings(task, tasks, rels)
        if siblings != None:
            generation = [task]
            generation.extend(siblings)
            path[-1] += str(chr(1+ord('@')+list(sorted(generation, key=lambda task: task.name)).index(task)))
        curr = find_parent(curr, tasks, rels)
    return '.'.join(reversed(path))

def gen_mermaid(tasks: list[Task], rels: list[NamedTuple]) -> str:
    mermaid = "graph LR\n"
    for task in tasks:
        mermaid += f'{slug(task.name)}[{task.name}<br/>{get_task_id(task, tasks, rels)}]'
        #if task.desc is not None:
        #    mermaid += f': {task.desc}'
        mermaid += '\n'
    
    for rel in rels:
        if isinstance(rel, BeforeAfter):
            mermaid += f'{slug(rel.before.name)} --> {slug(rel.after.name)}\n'
    
    return mermaid


if __name__ == '__main__':

    '''Phase 1'''
    project_setup = TaskWrapper(Task("Set up project"))

    social_elem = TaskWrapper(Task("Set up social elements"), parent=project_setup, previous=[project_setup])
    meet_team = TaskWrapper(Task("Meet team-mates"), parent=social_elem, previous=[social_elem])
    setup_comms = TaskWrapper(Task("Set up communication"), parent=social_elem, previous=[social_elem])
    dummy_comms = TaskWrapper(Task("Set up communication dummmy"), parent=social_elem, previous=[setup_comms])

    setup_organization = TaskWrapper(Task("Set up organization"), parent=project_setup, previous=[project_setup])

    setup_software = TaskWrapper(Task("Set up software"), parent=project_setup, previous=[setup_organization])

    project_plan = TaskWrapper(Task("Develop project plan"), previous=[project_setup, meet_team, dummy_comms, setup_software])
    tasks, rels = enumerate_most_tasks_and_rels(project_plan)

    from pprint import pprint
    tasks, rels = patch_sibling_rels(tasks, rels)
    print(gen_mermaid(list(set(tasks)), list(set(rels))))