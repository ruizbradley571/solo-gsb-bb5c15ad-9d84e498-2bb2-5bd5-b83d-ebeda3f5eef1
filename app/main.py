"""FastAPI 入口：单条诊断、批量诊断、修复建议与健康检查。"""

from __future__ import annotations

from typing import List

from fastapi import FastAPI

from .models import (
    BatchDiagnosisRequest,
    BatchDiagnosisResponse,
    BatchItemResponse,
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
from .solver import Observation, diagnose, suggest_repair

app = FastAPI(
    title="分布式时钟约束诊断 API",
    description=(
        "提交形如「B 时钟 - A 时钟 ∈ [min, max] 毫秒」的有向观测，"
        "服务将其转换为差分约束，判断时钟偏移是否一致；"
        "可行时给出相对参考节点的最紧上下界及对应的证据链"
        "（lower_proof / upper_proof），不可行时返回稳定的矛盾环，"
        "并可进一步给出最小观测删除集合的修复建议。"
    ),
    version="1.2.0",
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
