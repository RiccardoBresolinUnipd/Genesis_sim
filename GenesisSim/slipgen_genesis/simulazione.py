import genesis as gs
from slipgen.scene import setup_with_knobs
from slipgen.knobs import SlipKnobs

# 3. Carica il mondo (Robot, Tavolo, Cubi)
knobs = SlipKnobs(mu=0.5, fn_cap=10.0, disturb_level=0)
scene, franka, cam, end_eff, cubes, motors, fingers = setup_with_knobs(knobs, show_viewer=True)

# --- NOVITÀ: POSIZIONE DI DEFAULT ---
default_q = [0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78]
franka.control_dofs_position(default_q, motors) 

# 5. LOOP DI SIMULAZIONE
while True:
    # Qui potresti mettere i tuoi comandi di controllo, es:
    # franka.control_relative_positions(...)
    
    scene.step()