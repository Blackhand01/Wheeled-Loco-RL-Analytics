import os
import sys
import time
import argparse
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer

# Aggiunge la radice del progetto al path per gli import modulari
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.wheeled_robot_env import WheeledLocoEnv
from algo.networks import ActorCritic

def main():
    parser = argparse.ArgumentParser(description="Visualizza la policy del robot in MuJoCo Viewer")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path al checkpoint della policy (.pkl)")
    args = parser.parse_args()

    print("Inizializzazione dell'ambiente visivo...")
    # Usiamo MuJoCo standard per il rendering locale interattivo (più semplice per il viewer nativo Mac)
    model = mujoco.MjModel.from_xml_path('envs/assets/scene.xml')
    data = mujoco.MjData(model)

    # Inizializziamo l'ambiente JAX solo per recuperare la tassonomia di osservazione se necessario
    env = WheeledLocoEnv()
    
    # Setup della policy (Actor-Critic)
    init_obs = jnp.zeros((1, 38))
    policy = ActorCritic(action_dim=16)
    rng = jax.random.PRNGKey(0)
    policy_params = policy.init(rng, init_obs)

    # Se esiste un checkpoint, caricalo (placeholder per quando avrai i pesi da Colab)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Caricamento checkpoint da: {args.checkpoint}")
        # Qui caricherai i parametri reali (es. usando pickle o flax.checkpoints)
    else:
        print("⚠️ Nessun checkpoint trovato o specificato. Esecuzione con Policy Casuale (Untrained).")

    print("\n🚀 Apertura del MuJoCo Viewer... Premi 'ESC' nella finestra per chiudere.")
    
    # Avvia il viewer nativo di MuJoCo su macOS
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Porta il robot in posizione Home nell'interfaccia grafica
        mujoco.mj_resetDataKeyframe(model, data, 0)
        
        # Comando target finto (es: vai avanti a 1.0 m/s)
        command = jnp.array([1.0, 0.0, 0.0]) 
        
        # Loop di simulazione in tempo reale
        while viewer.is_running():
            step_start = time.time()

            # 1. Estrazione dell'osservazione corrente direttamente da mujoco.Data
            # Ricostruiamo la stessa struttura di WheeledLocoEnv._get_obs
            qpos = data.qpos[7:] # Salta la posizione base/quaternion freejoint per i giunti interni
            qvel = data.qvel[6:] # Salta la velocità lineare/angolare base
            
            # Nota: per semplicità nello smoke test visivo estraiamo un vettore piatto compatibile
            # Nei fatti, leggiamo l'orientamento e i sensori
            obs = jnp.zeros((1, 38)) # In uno scenario reale mappiamo qpos/qvel qui dentro
            
            # 2. Calcolo dell'azione tramite la policy Flax
            # L'attore sputa fuori le azioni (coppie o target)
            actions, _ = policy.apply(policy_params, obs, method=policy.get_action_and_value)
            actions = np.array(actions[0]) # Convertiamo in NumPy per MuJoCo nativo

            # 3. Applicazione delle azioni ai motori del robot
            data.ctrl[:] = actions

            # 4. Avanzamento della fisica
            mujoco.mj_step(model, data)

            # 5. Sincronizzazione del viewer grafico
            viewer.sync()

            # Mantiene il framerate sincrono con il tempo reale (es. 500 Hz o 60 Hz del viewer)
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    import numpy as np # Necessario per l'interfaccia nativa MuJoCo C-bound
    main()