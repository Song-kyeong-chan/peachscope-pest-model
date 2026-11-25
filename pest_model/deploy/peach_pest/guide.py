"""모델 클래스 → 대응방안(peach_pest_guide.json) 연결."""
import json
from pathlib import Path

TIER_ORDER = {"urgent": 3, "soon": 2, "next_season": 1, "none": 0}
# 그룹의 기본 등급 (질문으로 좁히기 전). 가이드 JSON display_groups[].response_window 설명과 같다.
GROUP_TIER = {"rot": "urgent", "spot": "soon", "damage": "soon"}


class Guide:
    def __init__(self, path: Path):
        self.data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.entries = {e["id"]: e for e in self.data["entries"]}
        self.groups = {g["id"]: g for g in self.data["display_groups"]}
        self.class_map = self.data["model_class_map"]
        self.version = self.data["_meta"]["version"]

    def resolve(self, label: str) -> tuple[str, str, str]:
        """모델 클래스 → (guide_id, guide_type, tier)."""
        legacy = self.class_map.get("_legacy_exp4_5", {})  # 실험 4·5 모델(6클래스) 출력도 계속 지원
        gid = self.class_map.get(label) or legacy.get(label, "unknown")
        if gid in self.groups:
            return gid, "group", GROUP_TIER.get(gid, "soon")
        e = self.entries.get(gid, self.entries["unknown"])
        return e["id"], "entry", e.get("response_window", {}).get("tier", "soon")

    def unknown(self) -> tuple[str, str, str]:
        e = self.entries["unknown"]
        return e["id"], "entry", e["response_window"]["tier"]

    def questions_for(self, probs: dict[str, float]) -> list[str]:
        """앱에서 물어볼 질문 규칙 id. spot이면 항상 spot_split, spot·insect가 비슷하면 spot_vs_insect."""
        top = sorted(probs, key=probs.get, reverse=True)
        q = []
        if top[0] == "spot":
            q.append("spot_split")
        if top[0] in ("damage", "insect_damage"):
            q.append("damage_split")
        if top[0] in ("spot", "damage", "insect_damage") and {top[0], top[1]} & {"spot"} and {top[0], top[1]} & {"damage", "insect_damage"} \
                and probs[top[0]] - probs[top[1]] < 0.2:
            q.append("spot_vs_damage")
        return q
