"""Model identities and the controls used by each movement rule."""
MODELS = {
    "couzin2002": "Zonal - Couzin et al. 2002 - 2D",
    "zonal_overlap": "Zonal - overlapping alignment and attraction - 2D",
    "couzin2005": "Zonal - Couzin et al. 2005 - 2D (uninformed)",
    "vicsek": "Vicsek et al. 1995 - 2D",
    "boids": "Boids - separation, alignment and cohesion - 2D",
    "voronoi": "Voronoi-filtered zonal - 2D",
}


def model_id(value):
    if value in {"boids_geopattern", "Boids - Reynolds-style steering - 2D",
                 "Boids - Huang steering implementation - 2D", "Boids - GeoPatternLab implementation - 2D"}:
        return "boids"
    if value == "zonal":  # Existing settings keep their extension behaviour.
        return "couzin2002"
    if value in MODELS:
        return value
    for key, label in MODELS.items():
        if value == label:
            return key
    raise ValueError(f"Unknown model {value!r}; choose from {', '.join(MODELS)}")


def zone_fields(value):
    model = model_id(value)
    if model == "vicsek":
        return {"r_a": "Interaction radius"}
    if model == "couzin2005":
        return {"r_r": "Repulsion radius", "r_a": "Alignment + attraction radius"}
    fields = {"r_r": "Repulsion radius", "r_o": "Alignment outer radius", "r_a": "Attraction outer radius"}
    if model == "zonal_overlap":
        fields["attraction_start"] = "Attraction inner radius"
    if model == "boids":
        fields = {"r_a": "Perception radius"}
    return fields
