"""inference algorithms for models based on Luce's choice axiom"""
# Simply import functions defined in modules.

from .lsr import (
    lsr_pairwise,
    ilsr_pairwise,
    lsr_rankings,
    ilsr_rankings,
    lsr_top1,
    ilsr_top1,
)

from .mm import (
    mm_pairwise,
)

from .utils import (
    footrule_dist,
    log_likelihood_pairwise,
    log_likelihood_rankings,
    log_likelihood_top1,
    generate_pairwise,
    generate_rankings,
    compare,
)

from .ep import ep_pairwise

from .opt import opt_pairwise
