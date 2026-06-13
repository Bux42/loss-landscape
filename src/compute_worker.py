"""Background compute worker thread."""

import threading
import queue
import numpy as np
from .physics import batch_simulate
from . import config


class ComputeWorker:
    """Background thread for batch physics simulations."""
    
    def __init__(
        self,
        landscape,
        world_height_cells,
        obstacles,
        friction,
        bounce_factor,
        gravity,
        velocity_threshold,
        dt,
        max_steps=1000,
    ):
        self.landscape = landscape
        self.world_height_cells = world_height_cells
        self.obstacles = obstacles
        self.friction = friction
        self.bounce_factor = bounce_factor
        self.gravity = gravity
        self.velocity_threshold = velocity_threshold
        self.dt = dt
        self.max_steps = max_steps
        
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.paused = False
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()  # Start unpaused

    def _drain_pending_tasks(self):
        """Remove all queued tasks that have not started yet."""
        while True:
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break
    
    def start(self):
        """Start worker thread."""
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop worker thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
    
    def pause(self):
        """Pause worker."""
        self.pause_event.clear()
        self.paused = True
    
    def resume(self):
        """Resume worker."""
        self.pause_event.set()
        self.paused = False
    
    def submit_task(self, start_positions, target_pos, task_id):
        """Submit computation task."""
        # New compute requests supersede old ones.
        self.cancel_current_task(clear_pending=True)
        self.task_queue.put((start_positions, target_pos, task_id))

    def cancel_current_task(self, clear_pending=True):
        """Cancel currently running compute and optionally clear queued computes."""
        self.cancel_event.set()
        if clear_pending:
            self._drain_pending_tasks()

    def set_bounce_factor(self, bounce_factor):
        """Update ball bounce factor used by future tasks."""
        self.bounce_factor = bounce_factor

    def set_simulation_params(
        self,
        friction=None,
        bounce_factor=None,
        gravity=None,
        velocity_threshold=None,
        dt=None,
        max_steps=None,
    ):
        """Update simulation parameters used by future tasks."""
        if friction is not None:
            self.friction = friction
        if bounce_factor is not None:
            self.bounce_factor = bounce_factor
        if gravity is not None:
            self.gravity = gravity
        if velocity_threshold is not None:
            self.velocity_threshold = velocity_threshold
        if dt is not None:
            self.dt = dt
        if max_steps is not None:
            self.max_steps = max_steps
    
    def get_result(self, block=False, timeout=None):
        """Get result if available."""
        try:
            return self.result_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def get_progress(self, block=False, timeout=None):
        """Get progress update if available."""
        try:
            return self.progress_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def _worker_loop(self):
        """Main worker loop."""
        while self.running:
            # Wait if paused
            self.pause_event.wait()
            
            try:
                start_pos, target_pos, task_id = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Fresh cancellation state for this specific task run.
            self.cancel_event.clear()

            circles = []
            for obs in self.obstacles:
                if obs.get("type") == "circle":
                    world_x = obs["x"] / (config.WINDOW_WIDTH - 1) * (config.LANDSCAPE_WIDTH - 1)
                    world_bottom_y = obs["y"] / (config.BOTTOM_PANEL_HEIGHT - 1) * (config.LANDSCAPE_HEIGHT - 1)
                    world_y = config.LANDSCAPE_HEIGHT + world_bottom_y
                    radius_world = obs["radius"] * (config.LANDSCAPE_WIDTH - 1) / (config.WINDOW_WIDTH - 1)
                    circles.append([
                        world_x,
                        world_y,
                        radius_world,
                    ])
            if circles:
                circle_obstacles = np.array(circles, dtype=np.float64)
            else:
                circle_obstacles = np.empty((0, 3), dtype=np.float64)
            ball_radius = config.BALL_RADIUS_SCREEN * (config.LANDSCAPE_WIDTH - 1) / (config.WINDOW_WIDTH - 1)
            
            total = len(start_pos)
            distances = np.zeros(total)
            # Larger chunks improve parallel kernel utilization and reduce scheduler overhead.
            chunk_size = max(1024, total // 8)

            # Run simulation in chunks and emit progress.
            done = 0
            cancelled = False
            while done < total:
                if self.cancel_event.is_set() or not self.running:
                    cancelled = True
                    break

                end = min(done + chunk_size, total)
                d_chunk = batch_simulate(
                    start_pos[done:end],
                    target_pos,
                    self.landscape,
                    circle_obstacles,
                    ball_radius,
                    self.world_height_cells,
                    self.friction,
                    self.bounce_factor,
                    self.gravity,
                    self.velocity_threshold,
                    self.dt,
                    self.max_steps,
                )
                distances[done:end] = d_chunk
                done = end
                self.progress_queue.put((task_id, done, total))

            # Enqueue completion/cancellation status.
            if cancelled:
                self.result_queue.put((task_id, None, True))
            else:
                self.result_queue.put((task_id, distances, False))
