using CairoMakie
using Printf

# 简支梁、跨中集中力 P 的相对 L2 误差（simple beam convergence P.jl）
n  = [10.0, 20.0, 40.0, 80.0]
eu = [1.488629e-02, 3.730674e-03, 9.331806e-04, 2.332711e-04]
ep = [1.000000e-02, 2.500000e-03, 6.250000e-04, 1.562500e-04]

function 拟合斜率(n, err)
    x = log10.(n);  y = log10.(err)
    x̄ = sum(x)/length(x);  ȳ = sum(y)/length(y)
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄).^2)
end

斜率u = 拟合斜率(n, eu)
斜率p = 拟合斜率(n, ep)

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
ax.title  = "简支梁（跨中集中力）"

text!(ax, 40, 1.8 * eu[3], text = @sprintf("kᵤ= %.2f", 斜率u),
      color = :dodgerblue, align = (:right, :bottom), fontsize = 14)
text!(ax, 40, 0.8 * ep[3], text = @sprintf("kᵩ= %.2f", 斜率p),
      color = :orange, align = (:right, :top), fontsize = 14)

axislegend(ax, position = :rt, framevisible = false)

save(joinpath(@__DIR__, "l2_error_simple_P.png"), fig, px_per_unit = 3)
save(joinpath(@__DIR__, "l2_error_simple_P.pdf"), fig)

println("已输出 l2_error_simple_P.png/pdf（简支梁，跨中集中力）")
