"""
Quick Visualization Script - Most Important Plots Only
Run this if the full visualize_tests.jl is having issues
"""

using DataDrivenAcoustics
using Plots
using Printf

println("="^80)
println("QUICK VISUALIZATION - CORE CONCEPTS")
println("="^80)

output_dir = "check/visualizations"
mkpath(output_dir)

# Helper function for rectangles
function rectangle(x1, y1, x2, y2)
    Shape([x1, x2, x2, x1], [y1, y1, y2, y2])
end

# ==============================================================================
# 1. PLANE WAVE PHYSICS - Show different angles
# ==============================================================================

println("\n[1/3] Plane Wave Physics...")

k = 2π * 100.0 / 1500.0  # wavenumber
x_range = range(-30, 30, length=100)
y_range = range(-30, 30, length=100)

# Four different angles
plots_pw = []
for (theta, label) in zip([0.0, π/4, π/2, 3π/4],
                          ["0° (East)", "45° (NE)", "90° (North)", "135° (NW)"])
    field = zeros(ComplexF64, length(y_range), length(x_range))

    for (i, y) in enumerate(y_range)
        for (j, x) in enumerate(x_range)
            field[i, j] = plane_wave_propagate(x, y, k, 1.0, 0.0, theta, 1e9)
        end
    end

    p = heatmap(x_range, y_range, real.(field),
               title="Ray from $label",
               xlabel="x (m)", ylabel="y (m)",
               c=:RdBu, clims=(-1, 1),
               aspect_ratio=:equal)

    # Add arrow
    arrow_len = 15
    quiver!([0], [0],
           quiver=([arrow_len*cos(theta)], [arrow_len*sin(theta)]),
           color=:green, linewidth=3)

    push!(plots_pw, p)
end

p1 = plot(plots_pw..., layout=(2,2), size=(1200, 1000))
savefig(p1, joinpath(output_dir, "1_plane_waves.png"))
println("  ✓ Saved: 1_plane_waves.png")

# ==============================================================================
# 2. SPHERICAL WAVE PHYSICS - Show 1/r decay
# ==============================================================================

println("\n[2/3] Spherical Wave Physics...")

k = 2π * 1000.0 / 1500.0
x_range = range(-40, 40, length=150)
y_range = range(-40, 40, length=150)

# Source at origin
field_sph = zeros(ComplexF64, length(y_range), length(x_range))

for (i, y) in enumerate(y_range)
    for (j, x) in enumerate(x_range)
        field_sph[i, j] = spherical_wave_propagate(x, y, 0.0, 0.0, k, 100.0, 0.0)
    end
end

p2a = heatmap(x_range, y_range, abs.(field_sph),
            title="Spherical Wave from Origin",
            xlabel="x (m)", ylabel="y (m)",
            c=:viridis, clims=(0, 5),
            aspect_ratio=:equal)
scatter!([0], [0], marker=:star, markersize=10, color=:red, label="Source")

# 1D decay profile
r_range = range(1, 100, length=200)
amplitude_decay = [100.0 / r for r in r_range]

p2b = plot(r_range, amplitude_decay,
          xlabel="Distance r (m)", ylabel="Amplitude |P|",
          title="1/r Decay Law",
          linewidth=3, color=:blue,
          label="A/r (A=100)",
          legend=:topright)

p2 = plot(p2a, p2b, layout=(1, 2), size=(1400, 600))
savefig(p2, joinpath(output_dir, "2_spherical_waves.png"))
println("  ✓ Saved: 2_spherical_waves.png")

# ==============================================================================
# 3. CONCEPT COMPARISON
# ==============================================================================

println("\n[3/3] Concept Summary...")

# Create a conceptual diagram
concept_plot = plot(
    legend=false,
    axis=false,
    ticks=false,
    xlims=(0, 10),
    ylims=(0, 10)
)

# Case 1 box
annotate!(2.5, 8, text("Case 1: PlaneWaveCurvModel", :center, 14, :bold))
annotate!(2.5, 7.2, text("Unknown Source Location", :center, 11))
annotate!(2.5, 6.5, text("Learn: θ (angle), d (curvature)", :center, 10))
annotate!(2.5, 5.8, text("Use: Far-field approximation", :center, 10))
plot!(rectangle(0.5, 3.5, 4, 8), fillalpha=0.1, fillcolor=:blue, linewidth=2)

# Case 2 box
annotate!(7.5, 8, text("Case 2: SphericalWaveModel", :center, 14, :bold))
annotate!(7.5, 7.2, text("Known Source Location", :center, 11))
annotate!(7.5, 6.5, text("Learn: A (amplitude), φ (phase)", :center, 10))
annotate!(7.5, 5.8, text("Use: 1/r decay physics", :center, 10))
plot!(rectangle(5.5, 3.5, 9.5, 8), fillalpha=0.1, fillcolor=:red, linewidth=2)

# Shared components
annotate!(5, 2.5, text("Shared: Ray Basis Neural Network (RBNN)", :center, 12, :bold))
annotate!(5, 1.8, text("Multiple rays (typically 60-100)", :center, 10))
annotate!(5, 1.2, text("Trained with Flux.jl + Automatic Differentiation", :center, 10))
plot!(rectangle(0.5, 0.5, 9.5, 3.2), fillalpha=0.05, fillcolor=:green, linewidth=2)

savefig(concept_plot, joinpath(output_dir, "3_concept_summary.png"))
println("  ✓ Saved: 3_concept_summary.png")

println("\n" * "="^80)
println("QUICK VISUALIZATION COMPLETE!")
println("Files saved to: $output_dir/")
println("="^80)
