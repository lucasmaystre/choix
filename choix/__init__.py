"""inference algorithms for models based on Luce's choice axiom"""
# Simply import functions defined in modules.

from .lsr import (
    lsr_pairwise,
    lsr_rankings,
    lsr_top1,
    ilsr_pairwise,
    ilsr_rankings,
    ilsr_top1,
)

from .mm import (
    mm_pairwise,
    mm_rankings,
    mm_top1,
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

from .opt import (
    opt_pairwise,
    opt_rankings,
    opt_top1,
)
