#ifndef GSE_LS_WRAPPEDCONSTRAINT_H
#define GSE_LS_WRAPPEDCONSTRAINT_H

#include "../GSEDefs.h"

GSE_NS_BEGIN

template <typename... Elems>
struct VarArgsToVector;

template <>
struct VarArgsToVector<>
{
	typedef boost::fusion::vector<> type;
};

template <typename Head, typename... Tail>
struct VarArgsToVector<Head, Tail...>
{
	typedef typename boost::fusion::result_of::push_back<typename VarArgsToVector<Tail...>::type,Head>::type type;
};

template<typename Constraint, typename... RVs>
struct WrappedConstraint : public Constraint
{
	typedef typename boost::fusion::result_of::as_vector<typename VarArgsToVector<RVs...>::type>::type ParamRVs;

	WrappedConstraint(const Constraint& c_, const ParamRVs& params_) : Constraint(c_), params(params_) {}

	const ParamRVs& getParams() const { return params; }
private:
	ParamRVs params;
};

template<typename Constraint, typename... Args>
WrappedConstraint<Constraint,Args...> wrapConstraint(const Constraint& c, Args... args)
{
	return {c,boost::fusion::make_vector(args...)};
}

GSE_NS_END

#endif
