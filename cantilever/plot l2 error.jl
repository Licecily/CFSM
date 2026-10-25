using CairoMakie
using Printf

# 悬臂梁、均布荷载 q 的相对 L2 误差（cantilever convergence.jl）
n  = [5.0, 10.0, 20.0, 40.0]
eu = [9.818990e-03, 2.506731e-03, 6.299372e-04, 1.576523e-04]
ep = [1.559487e-02, 3.932768e-03, 9.853088e-04, 2.464593e-04]

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
ax.title  = "悬臂梁（均布荷载）"

text!(ax, 20, 0.8 * eu[3], text = @sprintf("kᵤ= %.2f", slope_u),
      color = :dodgerblue, align = (:right, :top), fontsize = 14)
text!(ax, 20, 1.8 * ep[3], text = @sprintf("kᵩ= %.2f", slope_φ),
      color = :orange, align = (:right, :bottom), fontsize = 14)

axislegend(ax, position = :rt, framevisible = false)

save(joinpath(@__DIR__, "l2_error_cantilever.png"), fig, px_per_unit = 3)
save(joinpath(@__DIR__, "l2_error_cantilever.pdf"), fig)

println("已输出 l2_error_cantilever.png/pdf（悬臂梁，均布荷载）")
