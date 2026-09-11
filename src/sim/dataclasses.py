import genesis as gs
import genesis.vis.keybindings as kb



class World:
    '''
    A bundle of all the objects that make up a world and a single .step() function.
    '''
    def __init__(
        self,
        scene: gs.Scene,
        mobiles=None,
        landscape=None,
        agent=None,
    ):
        self.scene = scene
        self.mobiles = mobiles or {}
        self.landscape = landscape or {}
        self.agent = agent
        self.scene_features = None
        
    def step(self):
        self.scene.step()
 
        if self.agent is not None:
            self.agent.step()
    
        # do other needed updates like camera

#    def get_observations(self):
#        return self.scene.get_observations()
    
    def reset(self, randomize=False):
        self.scene.reset()

        if randomize:
            # randomize the scene
            pass

        if self.agent is not None:
            self.agent.reset(randomize=randomize)
    

def add_scene_controls(world):
    """Register keyboard controls for the Genesis viewer.

    For the moment, just a way to reset

    Args:
        world: contains a built gs.Scene instance with show_viewer=True.
    """

    # Press T for a normal reset
    world.scene.viewer.register_keybinds(
        kb.Keybind(
            name="reset",
            key=kb.Key.T,
            key_action=kb.KeyAction.PRESS,
            callback=world.reset,
            args=(False,)  # passes False to randomize parameter
        )
    )

    # Press Shift+T for a randomized reset
    world.scene.viewer.register_keybinds(
        kb.Keybind(
            name="shift_reset",
            key=kb.Key.T,
            key_mods=(kb.KeyMod.SHIFT,),
            key_action=kb.KeyAction.PRESS,
            callback=world.reset,
            args=(True,)   # passes True to randomize parameter
        )
    )
