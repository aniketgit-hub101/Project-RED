"""
Deterministic Fault-Injection Demo
Demonstrates control-flow micro-batch halving and gradient accumulation scaling upon simulated OOM.
Makes zero hardware benchmark, throughput, or memory reduction claims.
"""

class SimulatedEventBus:
    def __init__(self):
        self.micro_batch = 16
        self.grad_accum_steps = 1
        self.state = "RUNNING"

    def emit_fault(self, fault_type: str):
        if fault_type == "CUDA_OOM_SIMULATED":
            print(f"[RECOVERY] Caught {fault_type}. Halving micro-batch and scaling accumulation...")
            self.micro_batch //= 2
            self.grad_accum_steps *= 2
            self.state = "RECOVERED"

def main():
    bus = SimulatedEventBus()
    print(f"[INIT] Micro-batch: {bus.micro_batch}, Accumulation: {bus.grad_accum_steps}")
    
    # Inject synthetic memory spike
    bus.emit_fault("CUDA_OOM_SIMULATED")
    
    print(f"[STATE] Current status: {bus.state}")
    print(f"[STATE] Micro-batch: {bus.micro_batch}, Accumulation: {bus.grad_accum_steps}")
    assert bus.micro_batch == 8
    assert bus.grad_accum_steps == 2
    print("[SUCCESS] Fault recovery control-flow verified.")

if __name__ == "__main__":
    main()
