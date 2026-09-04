import panda_py

ROBOT_IP = "172.16.0.2"

def main():
    print(f"Connecting to Franka at {ROBOT_IP}")
    
    try:
        panda = panda_py.Panda(ROBOT_IP)
        print("Connected successfully")
        
        # print current state
        state = panda.get_state()
        print(f"Current joint positions: {state.q}")
        print(f"Current End Effector position:    {panda.get_position()}")
        
        # move to home
        print("Moving to home position...")
        panda.move_to_start()
        print("Robot is in home position")
        
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    main()