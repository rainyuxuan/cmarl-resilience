from magent2.environments import tiger_deer_v4, battle_v4, adversarial_pursuit_v4

envs_config = {
    "tiger_deer": {
        "module": tiger_deer_v4,
        "args": {
            "map_size": 30,
            "render_mode": "human",
            "max_cycles": 1000,
            "tiger_step_recover": 0,
            "deer_attacked": -0.1,
        }
    },
    "battle": {
        "module": battle_v4,
        "args": {
            "map_size": 35,
            "render_mode": "human",
            "max_cycles": 1000,
            "step_reward": -0.005,
            "attack_penalty": -0.05,
            "dead_penalty": -0.5,
            "attack_opponent_reward": 0.2,
        }
    },
    "adversarial_pursuit": {
        "module": adversarial_pursuit_v4,
        "args": {
            "map_size": 35,
            "render_mode": "human",
            "max_cycles": 1000,
            "tag_penalty": 0
        }
    },
    "adversarial_pursuit_big": {
        "module": adversarial_pursuit_v4,
        "args": {
            "map_size": 55,
            "render_mode": "human",
            "max_cycles": 1000,
            "tag_penalty": 0
        }
    },
    "adversarial_pursuit_small": {
        "module": adversarial_pursuit_v4,
        "args": {
            "map_size": 20,
            "render_mode": "human",
            "max_cycles": 1000,
            "tag_penalty": 0
        }
    }
}