# ADR-0002：普通客服不使用多Agent

状态：接受

普通客服请求要求低延迟和可预测执行，固定执行路由、检索、生成和校验。WorkBuddy等宿主只调用一次`answer_customer_question`，不得自行拆成多个实时工具调用。

