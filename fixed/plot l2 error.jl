using CairoMakie

# 相对 L2 误差数据（由 fixed beam convergence.jl 计算得到）
n  = [5.0, 10.0, 20.0, 40.0]
eu = [1.3588108302e-01, 3.5074296666e-02, 8.8372841559e-03, 2.2132226739e-03]
ep = [1.7977743781e-01, 4.5606805975e-02, 1.1442579346e-02, 2.8630435938e-03]

# 双对数坐标下最小二乘拟合的斜率
function 拟合斜率(n, err)
    x = log10.(n)
    y = log10.(err)
    x̄ = sum(x) / length(x)
    ȳ = sum(y) / length(y)
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄).^2)
end

斜率u = 拟合斜率(n, eu)
斜率p = 拟合斜率(n, ep)

CairoMakie.set_theme!(
    fonts = (; regular = "Microsoft YaHei", bold = "Microsoft YaHei",
               italic = "Microsoft YaHei", bolditalic = "Microsoft YaHei"),
    fontsize = 16)

# ---------- 位移 + 转角画在同一个坐标系 ----------
fig = Figure(size = (820, 560))
ax  = Axis(fig[1, 1])

# 先画原始数据，再设对数轴
scatterlines!(ax, n, eu, markersize = 11, linewidth = 2.2,
              color = :dodgerblue, label = "位移")
scatterlines!(ax, n, ep, markersize = 11, linewidth = 2.2,
              color = :orange, marker = :rect, label = "转角")

ax.xscale = log10
ax.yscale = log10
ax.xticks = n
ax.xlabel = "单元数   log₁₀(n)"
ax.ylabel = "误差     log₁₀(L₂)"
ax.title  = "固支梁（均布荷载）"

# 在两条曲线附近标注拟合斜率
text!(ax, 20, 8e-3, text = "kᵤ= " * string(round(斜率u, digits = 2)),
      color = :dodgerblue, align = (:right, :top), fontsize = 14)
text!(ax, 20, 18e-3, text = "kᵩ= " * string(round(斜率p, digits = 2)),
      color = :orange, align = (:right, :bottom), fontsize = 14)

axislegend(ax, position = :rt, framevisible = false)

save(joinpath(@__DIR__, "l2_error.png"), fig, px_per_unit = 3)
save(joinpath(@__DIR__, "l2_error.pdf"), fig)

println("已输出 l2_error.png/pdf（位移+转角同坐标系）")
