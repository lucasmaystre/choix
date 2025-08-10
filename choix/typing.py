from collections.abc import Sequence

PairwiseData = Sequence[tuple[int, int]]
RankingData = Sequence[tuple[int, ...]]
Top1Data = Sequence[tuple[int, tuple[int, ...]]]
