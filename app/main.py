"""FastAPI 入口：单条诊断、批量诊断、修复建议、校准计划与健康检查。"""

from __future__ import annotations

from typing import List

from fastapi import FastAPI

from .models import (
    BatchDiagnosisRequest,
    BatchDiagnosisResponse,
    BatchItemResponse,
    CalibrationPlanRequest,
    CalibrationPlanResponse,
    ContradictionCycle,
    CycleStep,
    DiagnosisRequest,
    DiagnosisResponse,
    FeasibleDiagnosis,
    InfeasibleDiagnosis,
    RepairRequest,
    RepairResponse,
    collect_nodes,
)
from .solver import (
    CandidateObservation,
    Observation,
    diagnose,
    plan_calibration,
    suggest_repair,
)

app = FastAPI(
    title="分布式时钟约束诊断 API",
    description=(
        "提交形如「B 时钟 - A 时钟 ∈ [min, max] 毫秒」的有向观测，"
        "服务将其转换为差分约束，判断时钟偏移是否一致；"
        "可行时给出相对参考节点的最紧上下界及对应的证据链"
        "（lower_proof / upper_proof），不可行时返回稳定的矛盾环，"
        "并可进一步给出最小观测删除集合的修复建议，"
        "或在候选校准观测中规划总成本最小的选用方案，"
        "使目标节点的偏移区间宽度全部达标。"
    ),
    version="1.3.0",
)


def _to_observation(raw) -> Observation:
    return Observation(
        id=raw.id,
        source=raw.source,
        target=raw.target,
        min=raw.min,
        max=raw.max,
    )


def _diagnosis_model(name, result):
    """把 solver 返回的诊断字典包装成响应模型。"""
    if result["feasible"]:
        return FeasibleDiagnosis(name=name, **result)

    cycle = result["cycle"]
    steps: List[CycleStep] = [
        CycleStep(
            observation_id=s["observation_id"],
            from_=s["from"],
            to=s["to"],
            constraint=s["constraint"],
            weight=s["weight"],
            bound=s["bound"],
        )
        for s in cycle["steps"]
    ]
    return InfeasibleDiagnosis(
        name=name,
        cycle=ContradictionCycle(
            steps=steps,
            accumulated_bound=cycle["accumulated_bound"],
            contradiction=cycle["contradiction"],
        ),
    )


def _run_diagnosis(request: DiagnosisRequest):
    result = diagnose(
        nodes=request.effective_nodes(),
        reference=request.reference,
        observations=[_to_observation(o) for o in request.observations],
    )
    return _diagnosis_model(request.name, result)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/diagnose", response_model=DiagnosisResponse, tags=["diagnosis"])
def diagnose_once(request: DiagnosisRequest) -> DiagnosisResponse:
    """提交一组观测与参考节点，返回单次诊断结果。"""
    return _run_diagnosis(request)


@app.post(
    "/api/v1/diagnose/batch",
    response_model=BatchDiagnosisResponse,
    tags=["diagnosis"],
)
def diagnose_batch(request: BatchDiagnosisRequest) -> BatchDiagnosisResponse:
    """批量诊断；结果按提交顺序（index）稳定返回。"""
    items = [
        BatchItemResponse(index=i, name=req.name, result=_run_diagnosis(req))
        for i, req in enumerate(request.diagnoses)
    ]
    return BatchDiagnosisResponse(count=len(items), results=items)


@app.post(
    "/api/v1/diagnose/repair",
    response_model=RepairResponse,
    tags=["diagnosis"],
)
def diagnose_repair(request: RepairRequest) -> RepairResponse:
    """修复建议：不可行时搜索最小观测删除集合，使剩余系统恢复可行。"""
    observations = [_to_observation(o) for o in request.observations]
    result = diagnose(
        nodes=request.effective_nodes(),
        reference=request.reference,
        observations=observations,
    )
    if result["feasible"]:
        # 系统本就可行：无需删除，建议为空
        return RepairResponse(
            name=request.name,
            max_repairs=request.max_repairs,
            removed_observation_ids=[],
            limit_reached=False,
            diagnosis=_diagnosis_model(request.name, result),
        )

    removed_ids, remaining = suggest_repair(
        nodes=request.effective_nodes(),
        reference=request.reference,
        observations=observations,
        max_repairs=request.max_repairs,
    )
    if removed_ids is None:
        # 上限内无可行修复：建议为空，回传原不可行诊断
        return RepairResponse(
            name=request.name,
            max_repairs=request.max_repairs,
            removed_observation_ids=[],
            limit_reached=True,
            diagnosis=_diagnosis_model(request.name, result),
        )

    # 修复后诊断：节点集合按「显式 nodes ∪ reference ∪ 剩余观测端点」重新计算
    repaired = diagnose(
        nodes=collect_nodes(request.nodes, request.reference, remaining),
        reference=request.reference,
        observations=remaining,
    )
    return RepairResponse(
        name=request.name,
        max_repairs=request.max_repairs,
        removed_observation_ids=removed_ids,
        limit_reached=False,
        diagnosis=_diagnosis_model(request.name, repaired),
    )


@app.post(
    "/api/v1/diagnose/calibration-plan",
    response_model=CalibrationPlanResponse,
    tags=["diagnosis"],
)
def diagnose_calibration_plan(
    request: CalibrationPlanRequest,
) -> CalibrationPlanResponse:
    """校准计划：选取总成本最小的候选观测子集，使目标节点宽度全部达标。"""
    base_observations = [_to_observation(o) for o in request.observations]
    nodes = request.effective_nodes()

    # 基础观测矛盾：直接回传现有矛盾环，不做任何选择
    base_result = diagnose(
        nodes=nodes, reference=request.reference, observations=base_observations
    )
    if not base_result["feasible"]:
        return CalibrationPlanResponse(
            name=request.name,
            status="base_infeasible",
            selected_ids=[],
            total_cost=None,
            diagnosis=_diagnosis_model(request.name, base_result),
        )

    candidates = [
        CandidateObservation(observation=_to_observation(c), cost=c.cost)
        for c in request.candidates
    ]
    targets = {t.node: t.max_width for t in request.targets}
    selected_ids, total_cost, selected = plan_calibration(
        nodes=nodes,
        reference=request.reference,
        observations=base_observations,
        candidates=candidates,
        targets=targets,
    )
    if selected_ids is None:
        # 无解：回传基础系统的可行诊断，便于对照差距
        return CalibrationPlanResponse(
            name=request.name,
            status="no_feasible_plan",
            selected_ids=[],
            total_cost=None,
            diagnosis=_diagnosis_model(request.name, base_result),
        )

    planned = diagnose(
        nodes=nodes,
        reference=request.reference,
        observations=base_observations + selected,
    )
    return CalibrationPlanResponse(
        name=request.name,
        status="ok",
        selected_ids=selected_ids,
        total_cost=total_cost,
        diagnosis=_diagnosis_model(request.name, planned),
    )
