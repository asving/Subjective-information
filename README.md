# Epiplexity

Working notes on computational information: an observer-dependent generalization of Shannon entropy, and its use in making "computation in physical processes" a well-posed question.

- **`epiplexity-reformulation.tex` / `.pdf`** — the technical draft, *Computational Information: A Generalization of Shannon Entropy*. Defines subjective entropy for a priced program class (an "observer"), with an explicit amortization parameter; develops the calculus (subadditivity, chain rule, Shearer) from closure axioms; proves a master theorem reducing every inequality of Shannon type to observer closure plus a normalization bound; derives a computationally bounded data-processing inequality; and situates epiplexity (Finzi et al. 2025) and predictive V-information (Xu et al. 2020) as the two axes of the framework.
- **`finding-computation.md`** — the surrounding discussion: where the framework came from (the implementation problem), what it buys (implementation surplus, gauge invariance, the structure–noise frontier), the two experimental arenas (mechanistic interpretability and mathematics itself), and the open problems (interventions, the internal observer, hierarchy, exchange rates).

- **`circuit-race.md`** — which circuit a network learns first, and when. Circuits are described as units, each a path of weights that multiply; the wait for a unit is derived from the weights still to grow, its correlation with the current error, and the optimiser (plain gradient descent against Adam). Checked on a toy transformer and on the leap staircase of Abbe, Boix-Adserà and Misiakiewicz.
- **`race/`** — the scripts that produce every number in `circuit-race.md`. Plain Python 3; `leap.py` needs numpy.

Latest compiled PDF also mirrored at [asving.com](https://asving.com/uploads/2026/01/epiplexity-reformulation.pdf).

**Companion document:** *Computational Platonism* ([arXiv:2506.19787](https://arxiv.org/abs/2506.19787), source in [Math-and-AI](https://github.com/asving/Math-and-AI)) — the philosophical side of this program: mathematics as the experimental study of computational systems. It supplies the direction-of-fit move that dissolves the classical triviality arguments, and its notion of mathematical "surprise" is what the bounded data-processing inequality formalizes.
