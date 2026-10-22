using CairoMakie
using Printf

# 悬臂梁、自由端集中力 P 的相对 L2 误差（cantilever convergence P.jl）
n  = [5.0, 10.0, 20.0, 40.0]
eu = [1.228207e-02, 3.097302e-03, 7.759600e-04, 1.940614e-04]
ep = [9.999930e-03, 2.499930e-03, 6.249300e-04, 1.561800e-04]

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
ax.title  = "悬臂梁（自由端集中力）"

text!(ax, 20, 2.0 * eu[3], text = @sprintf("kᵤ= %.2f", 斜率u),
      color = :dodgerblue, align = (:right, :top), fontsize = 14)
text!(ax, 20, 0.7 * ep[3], text = @sprintf("kᵩ= %.2f", 斜率p),
      color = :orange, align = (:right, :bottom), fontsize = 14)

axislegend(ax, position = :rt, framevisible = false)

save(joinpath(@__DIR__, "l2_error_cantilever_P.png"), fig, px_per_unit = 3)
save(joinpath(@__DIR__, "l2_error_cantilever_P.pdf"), fig)

println("已输出 l2_error_cantilever_P.png/pdf（悬臂梁，自由端集中力）")
