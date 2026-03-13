from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from music_latents.config_models import AverageGroupSpec, ConfigError, QuerySpec


def normalize_queries(raw_items: List[Any]) -> List[QuerySpec]:
    queries: List[QuerySpec] = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise ConfigError("Each query must be a mapping.")

        if "id" in item:
            queries.append(normalize_query(item, inherited_color=None))
            continue

        nested_queries = item.get("queries")
        if nested_queries is None:
            raise ConfigError(
                "Each queries item must be a query mapping or a query block with 'queries'."
            )
        if not isinstance(nested_queries, list) or not nested_queries:
            raise ConfigError("Query block 'queries' must be a non-empty list.")

        block_color = item.get("color")
        if block_color is not None and (
            not isinstance(block_color, str) or not block_color
        ):
            raise ConfigError("Query block 'color' must be a non-empty string.")

        for nested in nested_queries:
            queries.append(normalize_query(nested, inherited_color=block_color))

    return queries


def normalize_query(raw: Dict, inherited_color: Optional[str]) -> QuerySpec:
    if not isinstance(raw, dict):
        raise ConfigError("Each query must be a mapping.")

    query_id = raw.get("id")
    if not query_id or not isinstance(query_id, str):
        raise ConfigError("Each query must include a non-empty 'id'.")

    name = raw.get("name", query_id)
    if not isinstance(name, str) or not name:
        raise ConfigError(f"Query '{query_id}' field 'name' must be a string.")

    color = raw.get("color", inherited_color)
    if color is not None and (not isinstance(color, str) or not color):
        raise ConfigError(f"Query '{query_id}' field 'color' must be a string.")
    target = raw.get("target", raw.get("label"))

    input_fields: List[Tuple[str, Any]] = []
    for key in ("abc", "midi_path", "musicxml_path", "mei_path", "custom", "value"):
        if key in raw and raw[key] is not None:
            value = raw[key]
            if key == "value":
                if not isinstance(value, int):
                    raise ConfigError(
                        f"Query '{query_id}' field 'value' must be an integer in [1, 120]."
                    )
                if value < 1 or value > 120:
                    raise ConfigError(
                        f"Query '{query_id}' field 'value' must be in [1, 120]."
                    )
            else:
                if not isinstance(value, str):
                    raise ConfigError(
                        f"Query '{query_id}' field '{key}' must be a string."
                    )
            input_fields.append((key, value))

    if len(input_fields) != 1:
        raise ConfigError(
            f"Query '{query_id}' must define exactly one input field: abc, midi_path, musicxml_path, mei_path, custom, or value."
        )

    input_kind, value = input_fields[0]
    return QuerySpec(
        query_id=query_id,
        name=name,
        color=color,
        target=target,
        input_kind=input_kind,
        value=value,
    )


def normalize_averages(
    raw: Any,
    queries: List[QuerySpec],
    method: str = "average",
) -> List[AverageGroupSpec]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise ConfigError(f"'{method}' must be a mapping if provided.")

    query_map = {query.query_id: query for query in queries}
    groups: List[AverageGroupSpec] = []

    for group_id, group_raw in raw.items():
        if not isinstance(group_id, str) or not group_id:
            raise ConfigError("Average group ids must be non-empty strings.")
        if group_id in query_map:
            raise ConfigError(
                f"Average group id '{group_id}' conflicts with an existing query id."
            )

        if isinstance(group_raw, list):
            query_ids = group_raw
            name = group_id
            color = None
        elif isinstance(group_raw, dict):
            query_ids = group_raw.get("queries")
            name = group_raw.get("name", group_id)
            color = group_raw.get("color")
        else:
            raise ConfigError(
                f"Average group '{group_id}' must be a list or a mapping."
            )

        if not isinstance(query_ids, list) or not query_ids:
            raise ConfigError(
                f"Average group '{group_id}' must define a non-empty 'queries' list."
            )

        normalized_query_ids: List[str] = []
        for query_id in query_ids:
            if not isinstance(query_id, str) or not query_id:
                raise ConfigError(
                    f"Average group '{group_id}' contains an invalid query id."
                )
            if query_id not in query_map:
                raise ConfigError(
                    f"Average group '{group_id}' references unknown query '{query_id}'."
                )
            normalized_query_ids.append(query_id)

        if len(normalized_query_ids) != len(set(normalized_query_ids)):
            raise ConfigError(
                f"Average group '{group_id}' has duplicate query references."
            )

        if not isinstance(name, str) or not name:
            raise ConfigError(f"Average group '{group_id}' name must be a string.")
        if color is not None and (not isinstance(color, str) or not color):
            raise ConfigError(f"Average group '{group_id}' color must be a string.")

        if color is None:
            colors = {query_map[qid].color for qid in normalized_query_ids}
            if len(colors) == 1:
                color = next(iter(colors))

        groups.append(
            AverageGroupSpec(
                group_id=group_id,
                query_ids=normalized_query_ids,
                name=name,
                color=color,
                method=method,
            )
        )

    return groups
