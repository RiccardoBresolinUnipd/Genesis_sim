import genesis as gs
import numpy as np
import cv2
from openpi_client import websocket_client_policy
from slipgen.scene import setup_with_knobs
from slipgen.knobs import SlipKnobs
import time

# --- CONFIGURAZIONE ---
OPEN_LOOP_HORIZON = 8  # Numero di passi temporali da eseguire per ogni inferenza

physics_dt = 0.01  # Il passo di integrazione della simulazione (10ms)
substeps = 15
# Il vero delta temporale di ogni comando nell'orizzonte
actual_dt = physics_dt * substeps

def main():
    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
    knobs = SlipKnobs(mu=0.5, fn_cap=10.0, disturb_level=0)
    
    scene, franka, cam, cam_ext, cam_wrist, end_eff, cubes, motors, fingers = setup_with_knobs(knobs, show_viewer=True)

    default_q = [0.0, 0.2, 0.0, -1.5, 0.0, 1.8, 0.8]

    print("Spostamento in posizione di default...")
    for _ in range(100):
        franka.control_dofs_position(default_q, motors)
        scene.step()

    print("Default position reached")
    time.sleep(1)

    instruction = "move the cube, then come back to home position"

    try:
        while True:
            # --- A. CATTURA IMMAGINI ---
            rgb_ext = cam_ext.render()[0]
            rgb_wrist = cam.render()[0]
            
            # Conversione NumPy/CPU
            img_raw_ext = rgb_ext.cpu().numpy() if hasattr(rgb_ext, 'cpu') else rgb_ext
            img_raw_wrist = rgb_wrist.cpu().numpy() if hasattr(rgb_wrist, 'cpu') else rgb_wrist

            # Normalizzazione 0-255
            if img_raw_ext.max() <= 1.0: img_raw_ext = (img_raw_ext * 255).astype(np.uint8)
            if img_raw_wrist.max() <= 1.0: img_raw_wrist = (img_raw_wrist * 255).astype(np.uint8)
            
            img_vla_ext = cv2.resize(img_raw_ext, (224, 224))
            img_vla_wrist = cv2.resize(img_raw_wrist, (224, 224))
            
            cv2.imshow("VLA - Exterior", cv2.cvtColor(img_vla_ext, cv2.COLOR_RGB2BGR))
            cv2.imshow("VLA - Wrist", cv2.cvtColor(img_vla_wrist, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            # --- B. INFERENZA ---
            joints_state = franka.get_dofs_position().cpu().numpy()[:7]
            gripper_state = franka.get_dofs_position().cpu().numpy()[7:8]

            obs = {
                'observation/exterior_image_1_left': img_vla_ext, 
                'observation/wrist_image_left': img_vla_wrist,      
                'observation/joint_position': joints_state,
                'observation/gripper_position': gripper_state,
                'prompt': instruction,
            }

            result = client.infer(obs)
            actions = result["actions"] # Spesso ne restituisce più di 8, noi usiamo l'orizzonte definito

            print(f"Received actions: {actions}")
            time.sleep(1)  # Simula il tempo di elaborazione

            # --- C. ESECUZIONE OPEN-LOOP ---
            for i in range(OPEN_LOOP_HORIZON):
                # 1. Recupera lo stato attuale (NumPy su CPU)
                all_dofs = franka.get_dofs_position().cpu().numpy()
                curr_q = all_dofs[:7]
                curr_g = all_dofs[7]

                # 2. Estrai velocità dall'azione
                # Se l'azione è [v1, v2, v3, v4, v5, v6, v7, v_gripper]
                joint_velocities = actions[i, :7]
                gripper_velocity = actions[i, 7]
                
                # 3. INTEGRAZIONE: Calcolo Target Position
                # dt_sim è il tempo che intercorre tra un'azione e l'altra.
                # Se il modello sputa fuori velocità normalizzate, potresti doverle scalare.
                dt_step = actual_dt  # Usa il vero delta temporale

                target_q = curr_q + (joint_velocities * dt_step)
                target_g = curr_g + (gripper_velocity * dt_step)

                target_g = np.clip(target_g * 0.04, 0.0, 0.04)
                # 5. Costruzione comando (Assicuriamoci che sia lungo 9)
                # target_q (7) + gripper dita (2) = 9
                full_command = np.concatenate([target_q, [target_g, target_g]])
                
                # Trasformiamo in float32 per Genesis/Torch
                full_command = full_command.astype(np.float32)

                # 6. Invio al robot
                franka.control_dofs_position(full_command)

                # Step della simulazione per rendere il movimento fluido
                for _ in range(substeps): 
                    scene.step()

    except KeyboardInterrupt:
        print("Chiusura...")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()