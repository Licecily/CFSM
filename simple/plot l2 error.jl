using CairoMakie
using Printf

# 简支梁、均布荷载 q 的相对 L2 误差（simple beam convergence.jl）
n  = [5.0, 10.0, 20.0, 40.0]
eu = [8.975397e-02, 2.288159e-02, 5.748047e-03, 1.438646e-03]
ep = [6.701712e-02, 1.692197e-02, 4.240907e-03, 1.060877e-03]

function fit_slope(n, err)
    x = log10.(n);  y = log10.(err)
    x̄ = sum(x)/length(x);  ȳ = sum(y)/length(y)
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄).^2)
end

slope_u = fit_slope(n, eu)
slope_φ = fit_slope(n, ep)

CairoMakie.set_theme!(
    fonts = (; regular = "Microsoft YaHei", bold = "Microsoft YaHei",
               italic = "Microsoft YaHei", bolditalic = "Microsoft YaHei"),
    fontsize = 16)

fig = Figure(size = (820, 560))
ax  = Axis(fig[1, 1])

scatterlines!(ax, n, eu, markersize = 11, linewidth = 2.2,
              color = :dodgerblue, label = "位移")
scatterlines!(ax, n, ep, markersize = 11, linewidth = 2.2,
              color = :orange, marker = :rect, label = "转角")

ax.xscale = log10
ax.yscale = log10
ax.xticks = n
ax.xlabel = "单元数   log₁₀(n)"
ax.ylabel = "误差     log₁₀(L₂)"
ax.title  = "简支梁（均布荷载）"

text!(ax, 20, 1.8 * eu[3], text = @sprintf("kᵤ= %.2f", slope_u),
      color = :dodgerblue, align = (:right, :bottom), fontsize = 14)
text!(ax, 20, 0.8 * ep[3], text = @sprintf("kᵩ= %.2f", slope_φ),
      color = :orange, align = (:right, :top), fontsize = 14)

axislegend(ax, position = :rt, framevisible = false)

save(joinpath(@__DIR__, "l2_error_simple.png"), fig, px_per_unit = 3)
save(joinpath(@__DIR__, "l2_error_simple.pdf"), fig)

println("已输出 l2_error_simple.png/pdf（简支梁，均布荷载）")
