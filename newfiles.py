import abc
import datetime
import random
import time
import threading
from collections import deque, defaultdict
import heapq
import uuid

# --- 1. Abstract Base Classes (ABCs) for Core Components ---

class Resource(abc.ABC):
    """Abstract base class for any system resource."""
    def __init__(self, resource_id: str, capacity: int):
        self._resource_id = resource_id
        self._capacity = capacity
        self._current_load = 0
        self._is_active = True
        self._created_at = datetime.datetime.now()
        self._logs = deque(maxlen=100) # Simple log buffer

    @property
    def resource_id(self) -> str:
        return self._resource_id

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def current_load(self) -> int:
        return self._current_load

    @property
    def is_active(self) -> bool:
        return self._is_active

    def activate(self):
        self._is_active = True
        self._log_action(f"Resource activated.")

    def deactivate(self):
        self._is_active = False
        self._log_action(f"Resource deactivated.")

    def _log_action(self, message: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._logs.append(f"[{timestamp}] {self.resource_id}: {message}")

    def get_logs(self):
        return list(self._logs)

    @abc.abstractmethod
    def utilize(self, amount: int) -> bool:
        """Attempt to utilize the resource by a given amount."""
        pass

    @abc.abstractmethod
    def release(self, amount: int):
        """Release a given amount of resource utilization."""
        pass

    @abc.abstractmethod
    def status(self) -> str:
        """Returns a string representation of the resource's status."""
        pass

class Task(abc.ABC):
    """Abstract base class for any task to be processed."""
    def __init__(self, task_id: str, complexity: int, priority: int = 5):
        self._task_id = task_id
        self._complexity = max(1, complexity) # Ensure positive complexity
        self._priority = max(1, min(10, priority)) # Priority 1 (highest) to 10 (lowest)
        self._created_at = datetime.datetime.now()
        self._status = "PENDING"
        self._assigned_resource_id: str | None = None
        self._started_at: datetime.datetime | None = None
        self._completed_at: datetime.datetime | None = None

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def complexity(self) -> int:
        return self._complexity

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def status(self) -> str:
        return self._status

    @property
    def assigned_resource_id(self) -> str | None:
        return self._assigned_resource_id

    def set_assigned_resource(self, resource_id: str):
        if self._status == "PENDING":
            self._assigned_resource_id = resource_id
            self._status = "ASSIGNED"
            self._started_at = datetime.datetime.now()

    def complete(self):
        if self._status == "PROCESSING" or self._status == "ASSIGNED":
            self._status = "COMPLETED"
            self._completed_at = datetime.datetime.now()

    def fail(self, reason: str = "Unknown"):
        if self._status != "COMPLETED":
            self._status = f"FAILED: {reason}"
            self._completed_at = datetime.datetime.now() # Mark as failed at this time

    def __lt__(self, other):
        """Used for priority queueing: lower priority value means higher priority."""
        return self.priority < other.priority if isinstance(other, Task) else NotImplemented

    @abc.abstractmethod
    def execute(self, resource: 'Resource') -> bool:
        """Simulate task execution using the provided resource."""
        pass

# --- 2. Concrete Implementations of Resources ---

class CPUCore(Resource):
    """Represents a CPU core resource."""
    def __init__(self, core_id: str, speed_ghz: float = 2.5):
        super().__init__(core_id, capacity=int(speed_ghz * 1000)) # Capacity in MIPS (arbitrary unit)
        self._speed_ghz = speed_ghz
        self._log_action(f"CPUCore initialized with {speed_ghz} GHz speed.")

    def utilize(self, amount: int) -> bool:
        if not self.is_active:
            self._log_action(f"Attempted to utilize inactive CPUCore.")
            return False
        if self._current_load + amount <= self.capacity:
            self._current_load += amount
            self._log_action(f"Utilized {amount}. Current load: {self._current_load}/{self.capacity}")
            return True
        self._log_action(f"Failed to utilize {amount}. Overload detected ({self._current_load}/{self.capacity}).")
        return False

    def release(self, amount: int):
        self._current_load = max(0, self._current_load - amount)
        self._log_action(f"Released {amount}. Current load: {self._current_load}/{self.capacity}")

    def status(self) -> str:
        return (f"CPU Core '{self.resource_id}': Speed={self._speed_ghz} GHz, "
                f"Load={self._current_load}/{self.capacity} ({(self._current_load/self.capacity)*100:.2f}%), "
                f"Status={'Active' if self.is_active else 'Inactive'}")

class MemoryBlock(Resource):
    """Represents a block of memory resource."""
    def __init__(self, block_id: str, size_mb: int):
        super().__init__(block_id, capacity=size_mb) # Capacity in MB
        self._size_mb = size_mb
        self._log_action(f"MemoryBlock initialized with {size_mb} MB capacity.")

    def utilize(self, amount: int) -> bool:
        if not self.is_active:
            self._log_action(f"Attempted to utilize inactive MemoryBlock.")
            return False
        if self._current_load + amount <= self.capacity:
            self._current_load += amount
            self._log_action(f"Allocated {amount}MB. Current usage: {self._current_load}/{self.capacity}MB")
            return True
        self._log_action(f"Failed to allocate {amount}MB. Insufficient memory ({self._current_load}/{self.capacity}MB).")
        return False

    def release(self, amount: int):
        self._current_load = max(0, self._current_load - amount)
        self._log_action(f"Deallocated {amount}MB. Current usage: {self._current_load}/{self.capacity}MB")

    def status(self) -> str:
        return (f"Memory Block '{self.resource_id}': Size={self._size_mb} MB, "
                f"Used={self._current_load}/{self.capacity} MB ({(self._current_load/self.capacity)*100:.2f}%), "
                f"Status={'Active' if self.is_active else 'Inactive'}")

# --- 3. Concrete Implementations of Tasks ---

class DataProcessingTask(Task):
    """A task involving data processing on a CPU."""
    def __init__(self, task_id: str, data_size_mb: int, processing_intensity: int, priority: int = 5):
        # Complexity is derived from data size and processing intensity
        super().__init__(task_id, complexity=data_size_mb * processing_intensity, priority=priority)
        self._data_size_mb = data_size_mb
        self._processing_intensity = processing_intensity
        self._required_memory = max(1, data_size_mb // 10) # Simple heuristic for memory

    @property
    def required_memory(self) -> int:
        return self._required_memory

    def execute(self, resource: Resource) -> bool:
        if not isinstance(resource, CPUCore):
            self.fail("Incorrect resource type: Expected CPUCore")
            return False

        # Simulate resource utilization
        cpu_load = self.complexity // 10 # Example: 1/10th of complexity as CPU load
        if resource.utilize(cpu_load):
            self._status = "PROCESSING"
            time_to_process = self.complexity / resource.capacity * 0.1 # Simulate processing time
            time.sleep(min(1.0, time_to_process)) # Cap sleep for faster simulation
            resource.release(cpu_load)
            self.complete()
            return True
        else:
            self.fail(f"CPU overload during execution on {resource.resource_id}")
            return False

class NetworkTransferTask(Task):
    """A task simulating network data transfer."""
    def __init__(self, task_id: str, transfer_size_mb: int, target_ip: str, priority: int = 7):
        # Complexity based on transfer size
        super().__init__(task_id, complexity=transfer_size_mb * 5, priority=priority)
        self._transfer_size_mb = transfer_size_mb
        self._target_ip = target_ip
        self._bandwidth_usage = max(1, transfer_size_mb // 20) # Simulate bandwidth requirement

    @property
    def bandwidth_usage(self) -> int:
        return self._bandwidth_usage

    def execute(self, resource: Resource) -> bool:
        # For simplicity, we'll map NetworkTransferTask to a CPUCore in this example
        # In a real system, you'd have a 'NetworkInterface' resource type.
        if not isinstance(resource, CPUCore): # Using CPU as a proxy for network ops
            self.fail("Incorrect resource type: Expected CPUCore for network simulation")
            return False

        cpu_load = self.bandwidth_usage * 5 # Simulate CPU effort for network ops
        if resource.utilize(cpu_load):
            self._status = "PROCESSING"
            time_to_transfer = self._transfer_size_mb / 100.0 # Simulate transfer time
            time.sleep(min(0.5, time_to_transfer)) # Cap sleep
            resource.release(cpu_load)
            self.complete()
            return True
        else:
            self.fail(f"Resource contention for network task on {resource.resource_id}")
            return False

# --- 4. System Components and Orchestration ---

class ResourceManager:
    """Manages a pool of various resources."""
    def __init__(self):
        self._resources: dict[str, Resource] = {}
        self._lock = threading.Lock() # Protect resource pool

    def add_resource(self, resource: Resource):
        with self._lock:
            if resource.resource_id in self._resources:
                print(f"Warning: Resource with ID '{resource.resource_id}' already exists. Overwriting.")
            self._resources[resource.resource_id] = resource
            print(f"Added resource: {resource.resource_id}")

    def get_resource(self, resource_id: str) -> Resource | None:
        with self._lock:
            return self._resources.get(resource_id)

    def get_available_resources(self, resource_type: type[Resource] = Resource) -> list[Resource]:
        with self._lock:
            return [res for res in self._resources.values()
                    if res.is_active and isinstance(res, resource_type) and res.current_load < res.capacity]

    def get_resource_by_type(self, resource_type: type[Resource]) -> list[Resource]:
        with self._lock:
            return [res for res in self._resources.values() if isinstance(res, resource_type)]

    def get_all_resource_statuses(self) -> dict[str, str]:
        with self._lock:
            return {res_id: res.status() for res_id, res in self._resources.items()}

class TaskScheduler:
    """Schedules tasks based on priority and resource availability."""
    def __init__(self, resource_manager: ResourceManager):
        self._resource_manager = resource_manager
        self._task_queue: list[Task] = [] # Min-heap for priority queue (lower priority value is higher priority)
        self._processing_tasks: dict[str, Task] = {} # Tasks currently being processed
        self._completed_tasks: dict[str, Task] = {}
        self._failed_tasks: dict[str, Task] = {}
        self._scheduler_thread: threading.Thread | None = None
        self._running = False
        self._queue_lock = threading.Lock() # Protect task queue and processing dicts
        self._event = threading.Event() # To signal the scheduler thread

    def add_task(self, task: Task):
        with self._queue_lock:
            heapq.heappush(self._task_queue, task)
            print(f"Added task '{task.task_id}' (Priority: {task.priority}, Complexity: {task.complexity}) to queue.")
            self._event.set() # Signal the scheduler that there's a new task

    def start(self):
        if not self._running:
            self._running = True
            self._scheduler_thread = threading.Thread(target=self._run_scheduler, name="TaskSchedulerThread", daemon=True)
            self._scheduler_thread.start()
            print("Task Scheduler started.")

    def stop(self):
        if self._running:
            self._running = False
            self._event.set() # Wake up the thread to check _running flag
            if self._scheduler_thread and self._scheduler_thread.is_alive():
                self._scheduler_thread.join(timeout=2) # Wait for thread to finish
            print("Task Scheduler stopped.")

    def _run_scheduler(self):
        while self._running:
            self._event.clear() # Clear the event before waiting
            task_to_process: Task | None = None
            resource_for_task: Resource | None = None

            with self._queue_lock:
                if not self._task_queue:
                    pass # No tasks, just wait
                else:
                    # Peek the highest priority task without removing it yet
                    potential_task = self._task_queue[0]

                    # Try to find a suitable resource based on task type
                    if isinstance(potential_task, DataProcessingTask):
                        available_cpus = self._resource_manager.get_available_resources(CPUCore)
                        if available_cpus:
                            # Simple strategy: pick the least loaded CPU
                            resource_for_task = min(available_cpus, key=lambda c: c.current_load / c.capacity)
                            if potential_task.complexity // 10 > (resource_for_task.capacity - resource_for_task.current_load):
                                resource_for_task = None # Not enough capacity on chosen CPU
                        # In a more complex system, you'd also check MemoryBlock availability here.
                        # For simplicity, we assume enough memory for DataProcessingTask if CPU is free.
                    elif isinstance(potential_task, NetworkTransferTask):
                        available_cpus = self._resource_manager.get_available_resources(CPUCore)
                        if available_cpus:
                            resource_for_task = min(available_cpus, key=lambda c: c.current_load / c.capacity)
                            if potential_task.bandwidth_usage * 5 > (resource_for_task.capacity - resource_for_task.current_load):
                                resource_for_task = None
                    # Add more task types and resource matching here

                    if resource_for_task:
                        task_to_process = heapq.heappop(self._task_queue) # Actually pop it
                        task_to_process.set_assigned_resource(resource_for_task.resource_id)
                        self._processing_tasks[task_to_process.task_id] = task_to_process
                        print(f"Task '{task_to_process.task_id}' assigned to {resource_for_task.resource_id}.")

            if task_to_process and resource_for_task:
                # Execute task in a separate thread to not block the scheduler
                exec_thread = threading.Thread(
                    target=self._execute_task_wrapper,
                    args=(task_to_process, resource_for_task),
                    name=f"TaskExec-{task_to_process.task_id}",
                    daemon=True
                )
                exec_thread.start()
            else:
                self._event.wait(timeout=1.0) # Wait for a new task or for 1 second to re-check

    def _execute_task_wrapper(self, task: Task, resource: Resource):
        try:
            success = task.execute(resource)
            with self._queue_lock:
                self._processing_tasks.pop(task.task_id, None)
                if success:
                    self._completed_tasks[task.task_id] = task
                    print(f"Task '{task.task_id}' completed successfully on {resource.resource_id}.")
                else:
                    self._failed_tasks[task.task_id] = task
                    print(f"Task '{task.task_id}' failed on {resource.resource_id}. Reason: {task.status}")
        except Exception as e:
            task.fail(f"Unhandled exception during execution: {e}")
            with self._queue_lock:
                self._processing_tasks.pop(task.task_id, None)
                self._failed_tasks[task.task_id] = task
                print(f"Task '{task.task_id}' failed due to exception on {resource.resource_id}: {e}")
        finally:
            # Ensure resource is released if task failed or completed
            # Note: task.execute() is responsible for releasing, but a crash might prevent it.
            # A more robust system would have resource monitoring for orphaned loads.
            pass

    def get_queue_status(self) -> dict:
        with self._queue_lock:
            return {
                "queued": len(self._task_queue),
                "processing": len(self._processing_tasks),
                "completed": len(self._completed_tasks),
                "failed": len(self._failed_tasks)
            }

    def get_processing_task_details(self) -> list[dict]:
        with self._queue_lock:
            return [
                {"id": t.task_id, "status": t.status, "assigned_resource": t.assigned_resource_id}
                for t in self._processing_tasks.values()
            ]

# --- 5. Main Simulation Environment ---

class SimulationEnvironment:
    """Orchestrates the entire system simulation."""
    def __init__(self):
        self.resource_manager = ResourceManager()
        self.task_scheduler = TaskScheduler(self.resource_manager)
        self.simulation_running = False
        self._sim_thread: threading.Thread | None = None

    def setup_system(self):
        print("\n--- Setting up System Resources ---")
        self.resource_manager.add_resource(CPUCore(f"CPU-01", 3.0))
        self.resource_manager.add_resource(CPUCore(f"CPU-02", 2.8))
        self.resource_manager.add_resource(MemoryBlock(f"MEM-01", 4096))
        self.resource_manager.add_resource(MemoryBlock(f"MEM-02", 2048))

        print("\n--- Initial Resource Status ---")
        for res_id, status_str in self.resource_manager.get_all_resource_statuses().items():
            print(f"- {status_str}")

    def generate_tasks(self, num_tasks: int = 10):
        print(f"\n--- Generating {num_tasks} Random Tasks ---")
        for i in range(num_tasks):
            task_type = random.choice([DataProcessingTask, NetworkTransferTask])
            task_id = f"TASK-{uuid.uuid4().hex[:6]}"
            priority = random.randint(1, 10) # 1 (high) to 10 (low)

            if task_type == DataProcessingTask:
                data_size = random.randint(50, 500) # MB
                intensity = random.randint(1, 5)
                task = DataProcessingTask(task_id, data_size, intensity, priority)
            else: # NetworkTransferTask
                transfer_size = random.randint(100, 1000) # MB
                target_ip = f"192.168.1.{random.randint(1, 254)}"
                task = NetworkTransferTask(task_id, transfer_size, target_ip, priority)
            self.task_scheduler.add_task(task)
            time.sleep(0.1) # Stagger task creation

    def run_simulation(self, duration_seconds: int = 10):
        self.setup_system()
        self.task_scheduler.start()
        self.simulation_running = True
        self._sim_thread = threading.Thread(target=self._simulation_loop, args=(duration_seconds,), daemon=True)
        self._sim_thread.start()
        print(f"\n--- Simulation started for {duration_seconds} seconds ---")
        self._sim_thread.join() # Wait for the simulation to complete
        print("\n--- Simulation ended ---")
        self.task_scheduler.stop()

    def _simulation_loop(self, duration_seconds: int):
        start_time = time.time()
        task_generation_interval = 2 # Generate tasks every X seconds
        last_generation_time = start_time

        while time.time() - start_time < duration_seconds:
            if time.time() - last_generation_time > task_generation_interval:
                num_new_tasks = random.randint(1, 3)
                print(f"\n[SIM] Generating {num_new_tasks} new tasks...")
                self.generate_tasks(num_new_tasks)
                last_generation_time = time.time()

            self.print_status_snapshot()
            time.sleep(1) # Snapshot every second

        self.simulation_running = False

    def print_status_snapshot(self):
        print("\n--- Current System Snapshot ---")
        print("Resource Status:")
        for res_id, status_str in self.resource_manager.get_all_resource_statuses().items():
            print(f"  - {status_str}")

        queue_stats = self.task_scheduler.get_queue_status()
        print(f"Task Queue: Queued={queue_stats['queued']}, Processing={queue_stats['processing']}, "
              f"Completed={queue_stats['completed']}, Failed={queue_stats['failed']}")

        if queue_stats['processing'] > 0:
            print("Currently Processing Tasks:")
            for task_info in self.task_scheduler.get_processing_task_details():
                print(f"  - {task_info['id']} (Status: {task_info['status']}, Resource: {task_info['assigned_resource']})")

    def shutdown(self):
        if self._sim_thread and self._sim_thread.is_alive():
            self.simulation_running = False
            self._sim_thread.join(timeout=2)
        self.task_scheduler.stop()
        print("\nSystem Shutdown Complete.")


# --- Entry Point ---
if __name__ == "__main__":
    env = SimulationEnvironment()
    try:
        env.run_simulation(duration_seconds=20) # Run for 20 seconds
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        env.shutdown()
        # Display final results
        print("\n--- Final Simulation Summary ---")
        print("Resource Logs:")
        for res in env.resource_manager.get_resource_by_type(Resource):
            print(f"\nLogs for {res.resource_id}:")
            for log_entry in res.get_logs():
                print(f"  {log_entry}")

        final_stats = env.task_scheduler.get_queue_status()
        print(f"\nTotal Tasks Completed: {final_stats['completed']}")
        print(f"Total Tasks Failed: {final_stats['failed']}")
        print(f"Tasks Remaining in Queue: {final_stats['queued']}")
        print(f"Tasks Still Processing (may complete shortly after shutdown): {final_stats['processing']}")
