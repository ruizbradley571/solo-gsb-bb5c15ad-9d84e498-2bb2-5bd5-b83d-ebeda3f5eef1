"""差分约束求解器测试。"""

import itertools

from app.solver import Observation, diagnose, suggest_repair


def _obs(id_, source, target, lo, hi):
    return Observation(id=id_, source=source, target=target, min=lo, max=hi)


def _bounds(result):
    return {b["node"]: (b["lower"], b["upper"]) for b in result["node_bounds"]}


def test_feasible_chain_tightest_bounds():
    # o1: x_B - x_A ∈ [1, 5]，o2: x_C - x_B ∈ [2, 4]
    result = diagnose(
        nodes=["A", "B", "C"],
        reference="A",
        observations=[_obs("o1", "A", "B", 1, 5), _obs("o2", "B", "C", 2, 4)],
    )
    assert result["feasible"] is True
    bounds = _bounds(result)
    assert bounds["A"] == (0.0, 0.0)
    assert bounds["B"] == (1.0, 5.0)
    # x_C - x_A：最紧上界 5+4=9，最紧下界 1+2=3
    assert bounds["C"] == (3.0, 9.0)
    assert result["unconstrained_nodes"] == []


def test_parallel_paths_sharpen_bounds():
    # A->B ∈ [1, 10]，A->C ∈ [0, 8]，B->C ∈ [-2, 2]
    result = diagnose(
        nodes=["A", "B", "C"],
        reference="A",
        observations=[
            _obs("o1", "A", "B", 1, 10),
            _obs("o2", "A", "C", 0, 8),
            _obs("o3", "B", "C", -2, 2),
        ],
    )
    assert result["feasible"] is True
    bounds = _bounds(result)
    # x_C - x_A 上界取 min(8, 10+2=12) = 8；下界取 max(0, 1-2=-1) = 0
    assert bounds["C"] == (0.0, 8.0)
    # x_B - x_A 上界取 min(10, 8-(-2)=10)=10；下界取 max(1, 0-2=-2)=1
    assert bounds["B"] == (1.0, 10.0)


def test_unreachable_node_marked_unconstrained():
    # D 与参考节点 A 不在同一弱连通分量
    result = diagnose(
        nodes=["A", "B", "D"],
        reference="A",
        observations=[_obs("o1", "A", "B", 0, 3), _obs("o2", "D", "D", -1, 1)],
    )
    assert result["feasible"] is True
    by_node = {b["node"]: b for b in result["node_bounds"]}
    assert by_node["A"]["constrained"] is True
    assert by_node["B"]["constrained"] is True
    assert by_node["D"]["constrained"] is False
    assert by_node["D"]["lower"] is None
    assert by_node["D"]["upper"] is None
    assert result["unconstrained_nodes"] == ["D"]


def test_single_observation_gives_both_bounds():
    # 普通 [lo, hi] 观测同时贡献上界边与下界边，因此参考节点的邻居双向可达。
    result = diagnose(
        nodes=["A", "B"],
        reference="A",
        observations=[_obs("o1", "A", "B", 0, 5)],
    )
    assert result["feasible"] is True
    assert _bounds(result)["B"] == (0.0, 5.0)


def test_simple_two_edge_contradiction_cycle():
    # o1: x_B - x_A ∈ [0, 10]；o2: x_A - x_B ∈ [3, 4]
    # o2 的下界边为 A->B 权 -3，与 o1 的下界边 B->A 权 0 构成和为 -3 的负环。
    result = diagnose(
        nodes=["A", "B"],
        reference="A",
        observations=[_obs("o1", "A", "B", 0, 10), _obs("o2", "B", "A", 3, 4)],
    )
    assert result["feasible"] is False
    cycle = result["cycle"]
    ids = [s["observation_id"] for s in cycle["steps"]]
    assert set(ids) == {"o1", "o2"}
    assert cycle["accumulated_bound"] == -3.0
    assert cycle["contradiction"] == "0 <= -3.0"
    # 每一步都是闭环上的有向边
    for i, step in enumerate(cycle["steps"]):
        assert step["from"] == cycle["steps"][i - 1]["to"]


def test_contradiction_cycle_is_canonical_and_stable():
    # 三节点矛盾环，多次运行与不同提交顺序结果一致
    obs_a = _obs("obs-a", "X", "Y", 0, 1)   # Y-X ∈ [0,1]
    obs_b = _obs("obs-b", "Y", "Z", 0, 1)   # Z-Y ∈ [0,1]
    obs_c = _obs("obs-c", "Z", "X", 2, 5)   # X-Z ∈ [2,5] => Z-X <= -2
    # 环 X->Y(1) + Y->Z(1) + Z->X(-2) = 0 恰可行；收紧下界
    obs_c = _obs("obs-c", "Z", "X", 3, 5)   # Z-X <= -3 => 下界边环和 -3

    def run(order):
        return diagnose(
            nodes=["X", "Y", "Z"],
            reference="X",
            observations=[obs_a, obs_b, obs_c][::order],
        )

    r1 = run(1)
    r2 = run(-1)
    assert r1["feasible"] is False and r2["feasible"] is False
    key1 = [(s["observation_id"], s["constraint"], s["from"], s["to"]) for s in r1["cycle"]["steps"]]
    key2 = [(s["observation_id"], s["constraint"], s["from"], s["to"]) for s in r2["cycle"]["steps"]]
    assert key1 == key2
    # 下界边环：X->Z(-3) + Z->Y(0) + Y->X(0) = -3
    assert r1["cycle"]["accumulated_bound"] == -3.0
    # 规范化起点：环从起点节点字典序最小的边开始
    start_nodes = [s["from"] for s in r1["cycle"]["steps"]]
    assert start_nodes[0] == min(start_nodes)


def test_self_loop_contradiction():
    # x_A - x_A ∈ [1, 2]：下界边 A->A 权 -1，自环负环
    result = diagnose(
        nodes=["A"],
        reference="A",
        observations=[_obs("bad", "A", "A", 1, 2)],
    )
    assert result["feasible"] is False
    assert [s["observation_id"] for s in result["cycle"]["steps"]] == ["bad"]
    assert result["cycle"]["accumulated_bound"] == -1.0


def test_no_observations_only_reference():
    result = diagnose(nodes=["A"], reference="A", observations=[])
    assert result["feasible"] is True
    assert result["node_bounds"][0]["lower"] == 0.0
    assert result["node_bounds"][0]["upper"] == 0.0
    assert result["unconstrained_nodes"] == []


def test_zero_interval_is_feasible():
    result = diagnose(
        nodes=["A", "B"],
        reference="A",
        observations=[_obs("eq", "A", "B", 3, 3)],
    )
    assert result["feasible"] is True
    assert _bounds(result)["B"] == (3.0, 3.0)


def test_nodes_sorted_in_output():
    result = diagnose(
        nodes=["C", "A", "B"],
        reference="A",
        observations=[_obs("o1", "A", "B", 0, 1), _obs("o2", "B", "C", 0, 1)],
    )
    assert [b["node"] for b in result["node_bounds"]] == ["A", "B", "C"]


def test_regression_negative_cycle_with_converging_parallel_edges():
    # 负环由两条同端点观测的不同边组成（id01 下界边 + id03 上界边），
    # 且环外尾巴的传播会在第 V 轮仍触发更新。
    result = diagnose(
        nodes=["N0", "N1", "N2", "N3", "N4"],
        reference="N4",
        observations=[
            _obs("id00", "N1", "N4", 3, 7),
            _obs("id01", "N1", "N0", 4, 4),
            _obs("id02", "N2", "N3", -6, -1),
            _obs("id03", "N1", "N0", -2, 3),
            _obs("id04", "N3", "N1", -3, -1),
        ],
    )
    assert result["feasible"] is False
    cycle = result["cycle"]
    assert cycle["accumulated_bound"] == -1.0  # N1->N0 权 3 + N0->N1 权 -4
    for i, step in enumerate(cycle["steps"]):
        assert step["from"] == cycle["steps"][i - 1]["to"]


def test_regression_relaxed_edge_endpoint_off_chain():
    # 额外扫描中第一条松弛边的终点位于传播链远端，前驱链长度恰为 V；
    # 负环提取必须从松弛边的起点回卷。
    result = diagnose(
        nodes=["N0", "N1", "N2", "N3"],
        reference="N1",
        observations=[
            _obs("id00", "N1", "N0", 4, 7),
            _obs("id01", "N1", "N3", -1, 1),
            _obs("id02", "N3", "N2", -2, 2),
            _obs("id03", "N2", "N3", -3, 0),
            _obs("id04", "N2", "N3", 5, 10),
        ],
    )
    assert result["feasible"] is False
    cycle = result["cycle"]
    assert cycle["accumulated_bound"] == -5.0  # N2->N3 权 0 + N3->N2 权 -5
    for i, step in enumerate(cycle["steps"]):
        assert step["from"] == cycle["steps"][i - 1]["to"]


def test_suggest_repair_already_feasible_returns_empty():
    obs = [_obs("o1", "A", "B", 0, 5)]
    removed, remaining = suggest_repair(
        nodes=["A", "B"], reference="A", observations=obs, max_repairs=3
    )
    assert removed == []
    assert remaining == obs


def test_suggest_repair_single_deletion_picks_lex_smallest():
    # o1/o2 互相矛盾，删除任意一条均可行 -> 选字典序最小的 o1
    obs = [_obs("o1", "A", "B", 0, 10), _obs("o2", "B", "A", 3, 4)]
    removed, remaining = suggest_repair(
        nodes=["A", "B"], reference="A", observations=obs, max_repairs=3
    )
    assert removed == ["o1"]
    assert [o.id for o in remaining] == ["o2"]


def test_suggest_repair_two_independent_contradictions():
    # 两对独立矛盾 a1/a2 与 b1/b2：需各删一条 -> 字典序最小组合 [a1, b1]
    obs = [
        _obs("a1", "A", "B", 0, 10),
        _obs("a2", "B", "A", 3, 4),
        _obs("b1", "C", "D", 0, 10),
        _obs("b2", "D", "C", 3, 4),
    ]
    removed, remaining = suggest_repair(
        nodes=["A", "B", "C", "D"], reference="A", observations=obs, max_repairs=3
    )
    assert removed == ["a1", "b1"]
    assert [o.id for o in remaining] == ["a2", "b2"]


def test_suggest_repair_self_loop_plus_pair():
    # bad 自环矛盾 + c1/c2 对矛盾：最小修复为删 bad 与 c1/c2 之一
    obs = [
        _obs("bad", "A", "A", 1, 2),
        _obs("c1", "A", "B", 0, 10),
        _obs("c2", "B", "A", 3, 4),
    ]
    removed, remaining = suggest_repair(
        nodes=["A", "B"], reference="A", observations=obs, max_repairs=3
    )
    assert removed == ["bad", "c1"]
    assert [o.id for o in remaining] == ["c2"]


def test_suggest_repair_limit_reached_returns_none():
    # 两处独立矛盾至少需要删 2 条，上限 1 时无解
    obs = [
        _obs("a1", "A", "B", 0, 10),
        _obs("a2", "B", "A", 3, 4),
        _obs("b1", "C", "D", 0, 10),
        _obs("b2", "D", "C", 3, 4),
    ]
    removed, remaining = suggest_repair(
        nodes=["A", "B", "C", "D"], reference="A", observations=obs, max_repairs=1
    )
    assert removed is None and remaining is None


def test_suggest_repair_zero_budget():
    obs = [_obs("bad", "A", "A", 1, 2)]
    removed, remaining = suggest_repair(
        nodes=["A"], reference="A", observations=obs, max_repairs=0
    )
    assert removed is None and remaining is None


def test_suggest_repair_does_not_mutate_input():
    obs = [_obs("o1", "A", "B", 0, 10), _obs("o2", "B", "A", 3, 4)]
    snapshot = list(obs)
    suggest_repair(nodes=["A", "B"], reference="A", observations=obs, max_repairs=3)
    assert obs == snapshot  # 原列表与观测对象均未改变


def _proof_keys(chain):
    return [(s["observation_id"], s["constraint"], s["from"], s["to"])
            for s in chain]


def test_reference_node_has_empty_proofs():
    result = diagnose(
        nodes=["A", "B"],
        reference="A",
        observations=[_obs("o1", "A", "B", 0, 5)],
    )
    a = next(b for b in result["node_bounds"] if b["node"] == "A")
    assert a["lower_proof"] == [] and a["upper_proof"] == []


def test_proof_chains_sum_to_bounds_and_walk():
    # o1: x_B - x_A ∈ [1, 5]，o2: x_C - x_B ∈ [2, 4]
    result = diagnose(
        nodes=["A", "B", "C"],
        reference="A",
        observations=[_obs("o1", "A", "B", 1, 5), _obs("o2", "B", "C", 2, 4)],
    )
    by_node = {b["node"]: b for b in result["node_bounds"]}
    c = by_node["C"]
    # 上界链：参考节点 -> 目标，权重和 = upper
    assert _proof_keys(c["upper_proof"]) == [
        ("o1", "upper", "A", "B"),
        ("o2", "upper", "B", "C"),
    ]
    assert sum(s["weight"] for s in c["upper_proof"]) == c["upper"] == 9.0
    # 下界链：目标 -> 参考节点，权重和的相反数 = lower
    assert _proof_keys(c["lower_proof"]) == [
        ("o2", "lower", "C", "B"),
        ("o1", "lower", "B", "A"),
    ]
    assert sum(s["weight"] for s in c["lower_proof"]) == -c["lower"] == -3.0


def test_unreachable_node_proofs_are_none():
    result = diagnose(
        nodes=["A", "D"],
        reference="A",
        observations=[_obs("solo", "D", "D", -1, 1)],
    )
    d = next(b for b in result["node_bounds"] if b["node"] == "D")
    assert d["lower"] is None and d["lower_proof"] is None
    assert d["upper"] is None and d["upper_proof"] is None


def test_proof_tie_break_lexicographic_and_order_independent():
    # A->B->D 与 A->C->D 等权（上界 4）；证据链取四元组序列字典序最小者
    obs = [
        _obs("o1", "A", "B", 0, 2),
        _obs("o2", "A", "C", 0, 1),
        _obs("o3", "B", "D", 0, 2),
        _obs("o4", "C", "D", 0, 3),
    ]
    results = [
        diagnose(nodes=["A", "B", "C", "D"], reference="A", observations=list(perm))
        for perm in itertools.permutations(obs)
    ]
    first = results[0]["node_bounds"]
    for r in results[1:]:
        assert r["node_bounds"] == first  # 与提交顺序无关
    d = next(b for b in first if b["node"] == "D")
    assert _proof_keys(d["upper_proof"]) == [
        ("o1", "upper", "A", "B"),
        ("o3", "upper", "B", "D"),
    ]
    assert _proof_keys(d["lower_proof"]) == [
        ("o3", "lower", "D", "B"),
        ("o1", "lower", "B", "A"),
    ]


def test_proof_tie_break_decided_at_later_step():
    # 两条等权链共享首步 o1，字典序在第二步决胜
    result = diagnose(
        nodes=["A", "B", "C", "D", "E"],
        reference="A",
        observations=[
            _obs("o1", "A", "B", 0, 1),
            _obs("o2", "B", "C", 0, 1),
            _obs("o3", "B", "D", 0, 1),
            _obs("o4", "C", "E", 0, 1),
            _obs("o5", "D", "E", 0, 1),
        ],
    )
    e = next(b for b in result["node_bounds"] if b["node"] == "E")
    assert _proof_keys(e["upper_proof"]) == [
        ("o1", "upper", "A", "B"),
        ("o2", "upper", "B", "C"),
        ("o4", "upper", "C", "E"),
    ]


def test_proof_chain_stays_simple_around_zero_weight_cycle():
    # 零权环 A<->B：非简单走法 R->A->B->A->V 四元组序列更小，
    # 但证据链必须是简单链 R->A->V
    result = diagnose(
        nodes=["R", "A", "B", "V"],
        reference="R",
        observations=[
            _obs("z", "R", "A", 0, 0),
            _obs("a", "A", "B", 0, 0),
            _obs("b", "A", "V", 0, 0),
        ],
    )
    v = next(b for b in result["node_bounds"] if b["node"] == "V")
    assert _proof_keys(v["upper_proof"]) == [
        ("z", "upper", "R", "A"),
        ("b", "upper", "A", "V"),
    ]
