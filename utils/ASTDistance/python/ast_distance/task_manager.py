"""Task manager for coordinating swarm agents — transliterated from task_manager.hpp."""

import json
import fcntl
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class TaskStatus(Enum):
    """Task status for porting work items."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class FileLock:
    """RAII file lock for preventing race conditions.
    
    Uses flock() for advisory locking on Unix systems.
    """
    def __init__(self, path: str):
        self.fd = -1
        self.locked = False
        self.lock_path = path + ".lock"
        
        try:
            self.fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o666)
            if self.fd >= 0:
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_EX)
                    self.locked = True
                except (OSError, IOError):
                    pass
        except (OSError, IOError):
            pass

    def __del__(self):
        if self.fd >= 0:
            if self.locked:
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
                except (OSError, IOError):
                    pass
            try:
                os.close(self.fd)
            except (OSError, IOError):
                pass

    def is_locked(self) -> bool:
        return self.locked


class PortTask:
    """A single porting task."""
    def __init__(self):
        self.source_path: str = ""
        self.source_qualified: str = ""
        self.target_path: str = ""
        self.target_qualified: str = ""
        self.dependent_count: int = 0
        self.dependency_count: int = 0
        self.status: TaskStatus = TaskStatus.PENDING
        self.assigned_to: str = ""
        self.assigned_at: str = ""
        self.completed_at: str = ""
        self.similarity: float = 0.0
        self.dependencies: List[str] = []
        self.dependents: List[str] = []


class TaskManager:
    """Task file manager for coordinating swarm agents."""
    
    def __init__(self, task_file: str):
        self.task_file_path: str = task_file
        self.agents_md_path: str = ""
        self.source_root: str = ""
        self.target_root: str = ""
        self.source_lang: str = ""
        self.target_lang: str = ""
        self.tasks: List[PortTask] = []

    def load(self) -> bool:
        """Load tasks from JSON file."""
        try:
            with open(self.task_file_path, 'r') as f:
                content = f.read()
        except Exception:
            return False

        self.tasks = []

        def extract_string(content: str, key: str) -> str:
            pattern = '"' + key + '"'
            pos = content.find(pattern)
            if pos == -1:
                return ""
            pos = content.find(':', pos)
            if pos == -1:
                return ""
            pos = content.find('"', pos)
            if pos == -1:
                return ""
            end = content.find('"', pos + 1)
            if end == -1:
                return ""
            return content[pos + 1:end]

        def extract_int(content: str, key: str) -> int:
            pattern = '"' + key + '"'
            pos = content.find(pattern)
            if pos == -1:
                return 0
            pos = content.find(':', pos)
            if pos == -1:
                return 0
            pos += 1
            while pos < len(content) and content[pos].isspace():
                pos += 1
            num = ""
            while pos < len(content) and content[pos].isdigit():
                num += content[pos]
                pos += 1
            return int(num) if num else 0

        self.source_root = extract_string(content, "source_root")
        self.target_root = extract_string(content, "target_root")
        self.source_lang = extract_string(content, "source_lang")
        self.target_lang = extract_string(content, "target_lang")
        self.agents_md_path = extract_string(content, "agents_md")

        tasks_pos = content.find('"tasks"')
        if tasks_pos == -1:
            return True

        pos = tasks_pos
        while True:
            start = content.find('{', pos)
            if start == -1:
                break
            end = content.find('}', start)
            if end == -1:
                break

            task_str = content[start:end + 1]

            if "source_path" not in task_str:
                pos = end + 1
                continue

            task = PortTask()
            task.source_path = extract_string(task_str, "source_path")
            task.source_qualified = extract_string(task_str, "source_qualified")
            task.target_path = extract_string(task_str, "target_path")
            task.dependent_count = extract_int(task_str, "dependent_count")
            task.assigned_to = extract_string(task_str, "assigned_to")
            task.assigned_at = extract_string(task_str, "assigned_at")
            task.completed_at = extract_string(task_str, "completed_at")

            status_str = extract_string(task_str, "status")
            if status_str == "pending":
                task.status = TaskStatus.PENDING
            elif status_str == "assigned":
                task.status = TaskStatus.ASSIGNED
            elif status_str == "completed":
                task.status = TaskStatus.COMPLETED
            elif status_str == "blocked":
                task.status = TaskStatus.BLOCKED

            if task.source_path:
                self.tasks.append(task)

            pos = end + 1

        return True

    def save(self) -> bool:
        """Save tasks to JSON file."""
        try:
            with open(self.task_file_path, 'w') as f:
                f.write('{\n')
                f.write(f'  "source_root": "{self.source_root}",\n')
                f.write(f'  "target_root": "{self.target_root}",\n')
                f.write(f'  "source_lang": "{self.source_lang}",\n')
                f.write(f'  "target_lang": "{self.target_lang}",\n')
                f.write(f'  "agents_md": "{self.agents_md_path}",\n')
                f.write('  "tasks": [\n')

                for i, t in enumerate(self.tasks):
                    f.write('    {\n')
                    f.write(f'      "source_path": "{t.source_path}",\n')
                    f.write(f'      "source_qualified": "{t.source_qualified}",\n')
                    f.write(f'      "target_path": "{t.target_path}",\n')
                    f.write(f'      "dependent_count": {t.dependent_count},\n')
                    f.write(f'      "status": "{self._status_to_string(t.status)}"')
                    if t.assigned_to:
                        f.write(f',\n      "assigned_to": "{t.assigned_to}"')
                    if t.assigned_at:
                        f.write(f',\n      "assigned_at": "{t.assigned_at}"')
                    if t.completed_at:
                        f.write(f',\n      "completed_at": "{t.completed_at}"')
                    f.write('\n    }')
                    if i < len(self.tasks) - 1:
                        f.write(',')
                    f.write('\n')

                f.write('  ]\n')
                f.write('}\n')
            return True
        except Exception:
            return False

    def assign_next(self, agent_id: str) -> Optional[PortTask]:
        """Assign the highest-priority pending task to an agent.
        
        Returns None if no tasks available.
        
        THREAD-SAFE: Uses file locking to prevent race conditions when
        multiple agents try to grab tasks simultaneously.
        """
        lock = FileLock(self.task_file_path)
        if not lock.is_locked():
            import sys
            print("Warning: Could not acquire lock on task file", file=sys.stderr)
            return None

        if not self.load():
            import sys
            print("Warning: Could not reload task file", file=sys.stderr)
            return None

        pending = [t for t in self.tasks if t.status == TaskStatus.PENDING]

        if not pending:
            return None

        pending.sort(key=lambda t: t.dependent_count, reverse=True)

        task = pending[0]
        task.status = TaskStatus.ASSIGNED
        task.assigned_to = agent_id
        task.assigned_at = self._current_timestamp()

        if not self.save():
            import sys
            print("Warning: Could not save task file after assignment", file=sys.stderr)
            task.status = TaskStatus.PENDING
            task.assigned_to = ""
            task.assigned_at = ""
            return None

        return task

    def complete_task(self, source_qualified: str) -> bool:
        """Mark a task as completed.
        
        THREAD-SAFE: Uses file locking.
        """
        lock = FileLock(self.task_file_path)
        if not lock.is_locked():
            import sys
            print("Warning: Could not acquire lock for complete_task", file=sys.stderr)
            return False

        if not self.load():
            return False

        for t in self.tasks:
            if t.source_qualified == source_qualified:
                t.status = TaskStatus.COMPLETED
                t.completed_at = self._current_timestamp()
                return self.save()

        return False

    def release_task(self, source_qualified: str) -> bool:
        """Release an assigned task back to pending.
        
        THREAD-SAFE: Uses file locking.
        """
        lock = FileLock(self.task_file_path)
        if not lock.is_locked():
            import sys
            print("Warning: Could not acquire lock for release_task", file=sys.stderr)
            return False

        if not self.load():
            return False

        for t in self.tasks:
            if t.source_qualified == source_qualified and t.status == TaskStatus.ASSIGNED:
                t.status = TaskStatus.PENDING
                t.assigned_to = ""
                t.assigned_at = ""
                return self.save()

        return False

    def get_stats(self) -> Tuple[int, int, int, int]:
        """Get task statistics."""
        pending = assigned = completed = blocked = 0
        for t in self.tasks:
            if t.status == TaskStatus.PENDING:
                pending += 1
            elif t.status == TaskStatus.ASSIGNED:
                assigned += 1
            elif t.status == TaskStatus.COMPLETED:
                completed += 1
            elif t.status == TaskStatus.BLOCKED:
                blocked += 1
        return pending, assigned, completed, blocked

    def read_agents_md(self) -> str:
        """Read AGENTS.md content if it exists."""
        if not self.agents_md_path:
            return ""
        try:
            with open(self.agents_md_path, 'r') as f:
                return f.read()
        except Exception:
            return ""

    def print_assignment(self, task: PortTask, agent_number: int) -> None:
        """Print task assignment details for an agent."""
        print("=== TASK ASSIGNMENT ===\n")
        print(f"You are agent #{agent_number}")
        print(f"Reminder: all ast_distance commands require: --agent {agent_number}\n")

        print("Source File:")
        print(f"  Path:      {self.source_root}/{task.source_path}")
        print(f"  Qualified: {task.source_qualified}")
        print(f"  Dependents: {task.dependent_count} files depend on this\n")

        print("Target File:")
        print(f"  Path:      {self.target_root}/{task.target_path}")
        prefix = "##" if self.target_lang == "python" else "//"
        print(f"  Add header: {prefix} port-lint: source {task.source_path}\n")

        print(f"Priority: {task.dependent_count} (higher = more critical)\n")

        agents_content = self.read_agents_md()
        if agents_content:
            print("=== PORTING GUIDELINES (from AGENTS.md) ===\n")
            print(agents_content + "\n")

        print("=== INSTRUCTIONS ===\n")
        print("1. Read the source file thoroughly")
        print("2. Create the target file at the target path")
        print("3. Add the port-lint header as the first line")
        print(f"4. Transliterate the source code to idiomatic {self.target_lang}")
        print("5. Match documentation comments from the source")
        print(f"6. Run: ast_distance --agent {agent_number} <source> {self.source_lang} <target> {self.target_lang}")
        print(f"   to verify similarity (aim for >0.85)")
        print(f"7. When complete, run: ast_distance --agent {agent_number} --complete {self.task_file_path} {task.source_qualified}\n")

    @staticmethod
    def _port_lint_comment_prefix(lang: str) -> str:
        """Use a visually-distinct prefix for Python."""
        if lang == "python":
            return "##"
        return "//"

    @staticmethod
    def _status_to_string(s: TaskStatus) -> str:
        return s.value

    @staticmethod
    def _current_timestamp() -> str:
        return datetime.now().isoformat()
