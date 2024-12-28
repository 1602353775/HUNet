import trace
# 创建一个Trace对象
tracer = trace.Trace(
    tracedirs=[sys.prefix, sys.exec_prefix], # 限制跟踪到Python标准库和第三方库
    trace=0, # 不打印执行的行
    count=1 # 开启计数模式
)

# 运行脚本
tracer.run('exec(open("test_pth.py").read())')

# 获取报告
results = tracer.results()
results.write_results(show_missing=True, summary=True, coverdir=".")
