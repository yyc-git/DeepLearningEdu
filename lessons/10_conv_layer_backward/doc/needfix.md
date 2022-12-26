# 命名不统一

- 课程中的图和描述为sensitivity map， 代码中却称为deltaMap





# TODO

- _paddingDeltaMap -> padding(为了进行full卷积而进行padding)为什么这样计算: (inputWidth + filterWidth - 1 - expandDeltaWidth) / 2   ？