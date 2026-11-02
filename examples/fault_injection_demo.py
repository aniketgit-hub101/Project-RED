from red.state import RedSharedState, RedState
from red.config import RedConfig
from red.engines.dbc import DynamicBatchController, RedVRAMExhaustedError

def main():
    config = RedConfig(min_batch_size=4)
    state = RedSharedState()
    state.state_machine.transition_to(RedState.RUNNING)
    
    dbc = DynamicBatchController(config, state)
    dbc._current_batch_size = 16
    print(f"[INIT] Micro-batch: {dbc._current_batch_size}")
    
    # Fault 1
    dbc.handle_oom()
    print(f"[RECOVERY 1] Adjusted micro-batch: {dbc._current_batch_size}")
    
    # Fault 2
    dbc.handle_oom()
    print(f"[RECOVERY 2] Adjusted micro-batch: {dbc._current_batch_size}")
    
    # Fault 3 (Hits min-batch limit)
    try:
        dbc.handle_oom()
    except RedVRAMExhaustedError as e:
        print(f"[TERMINAL FAULT] Caught expected exception: {e}")
        final_state = getattr(state.state_machine, "state", getattr(state.state_machine, "current_state", None))
        print(f"[STATE] Final state: {final_state}")

if __name__ == "__main__":
    main()