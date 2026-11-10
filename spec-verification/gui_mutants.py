import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gui_model as g
which = sys.argv[1]
if which == "need":        # 5.2 rule off by one
    orig = g.min_ctrl; g.min_ctrl = lambda o, p: orig(o, p) + 1
elif which == "cap":       # 19.7 step 4: degree cap without the -1
    src = g.Model.resolve
    def resolve(self, changed):
        src(self, changed); self.degree.set_sweep_maximum(max(self.count.spin.v, self.degree.spin.lo)) if self.count_used() and not self.count.active else None
    g.Model.resolve = resolve
elif which == "label":     # 19.6: label text swapped
    f = g.Model.flags
    g.Model.flags = lambda self: {**f(self), "count label": "Control points"}
# Exit status 0 when the mutant is detected (the model and the window disagree)
sys.exit(0 if g.main(600, seed=11) else 1)
