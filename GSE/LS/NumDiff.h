#ifndef GSE_LS_NUMDIFF_H
#define GSE_LS_NUMDIFF_H

#include "../GSEDefs.h"
#include <unsupported/Eigen/NumericalDiff>

GSE_NS_BEGIN

template<typename Constraint>
class EigenFunctorAdapter
{
public:
	typedef ConstraintTraits<Constraint> CTraits;
	enum { InputsAtCompileTime=CTraits::TotalParamDeltaDim };
	enum { ValuesAtCompileTime=CTraits::RetDim };
	typedef double Scalar;
	typedef Matrix<Scalar, InputsAtCompileTime, 1> InputType;
	typedef Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
	typedef Matrix<Scalar, ValuesAtCompileTime,InputsAtCompileTime> JacobianType;
	int inputs() const { return InputsAtCompileTime; }

	EigenFunctorAdapter(const Constraint& c_,const typename CTraits::ParamVecType& actualParams_) : constraint(c_), actualParams(actualParams_) {}

	typedef double T;

	struct MakeNullPointer
	{
		template <class T> struct result;
		template <class T> struct result<MakeNullPointer(const T&)>
		{
			typedef T* type;
		};

		template<typename T>
		T* operator()(const T& t) const
		{
			(void)t;
			return nullptr;
		}
	};

#if 0
	struct GetParamFromVector
	{
		GetParamFromVector(const Matrix<T,InputsAtCompileTime,1>& in_) : in(in_), offs(0) {}

		template <class T> struct result;
		template <class T> struct result<GetParamFromVector(const T&)>
		{
			typedef T type;
		};

		template<typename T>
		T operator()(const T&) const
		{
			T ret=in.template segment<T::RowsAtCompileTime>(offs);
			offs+=T::RowsAtCompileTime;
			return ret;
		}
		const Matrix<T,InputsAtCompileTime,1>& in;
		mutable int offs;
	};
#endif
	struct ForEachParam
	{
		ForEachParam(typename CTraits::ParamVecType& params_, const Matrix<T,InputsAtCompileTime,1>& in_) : params(params_), in(in_), offs(0) {}
		template<typename T> void operator()(T) const
		{
			const int Index=T::value;
			typedef typename boost::fusion::result_of::at_c<typename CTraits::ParamType,Index>::type RVType;
			typedef typename boost::remove_reference<RVType>::type::ManifoldType MType; // WHY IS THIS REMOVE REF NEEDED???
			boost::fusion::at_c<Index>(params)=MType::manifoldAdd(boost::fusion::at_c<Index>(params),in.template segment<MType::DeltaDim>(offs));
			offs+=MType::DeltaDim;
		}
		typename CTraits::ParamVecType& params;
		const Matrix<T,InputsAtCompileTime,1>& in;
		mutable int offs;
	};


	//template<typename T>
	void operator()(const Matrix<T,InputsAtCompileTime,1>& in, Matrix<T,ValuesAtCompileTime,1>* v) const
	{
		typename CTraits::ParamVecType params=actualParams;
		for_range<0,CTraits::NumParams>(ForEachParam(params,in));

		typename CTraits::ParamJacType jac;

		auto jacptr=boost::fusion::transform(jac,MakeNullPointer());

		auto fnParams=boost::fusion::push_front(boost::fusion::join(params,jacptr),&constraint);
		*v=boost::fusion::invoke(&Constraint::error,fnParams);

	}

	// for compatibility with NumericalDiff
	int values() const { return ValuesAtCompileTime; }
	//template<typename T>
	void operator()(const Matrix<T,InputsAtCompileTime,1>& in, Matrix<T,ValuesAtCompileTime,1>& v) const
	{
		//this->operator ()<T>(in,&v);
		this->operator ()(in,&v);
	}
private:
	const Constraint& constraint;
	typename CTraits::ParamVecType actualParams;
};

template<typename Constraint>
struct DissectJacobian
{
	typedef ConstraintTraits<Constraint> CTraits;
	DissectJacobian(const Matrix<double,CTraits::RetDim,CTraits::TotalParamDeltaDim>& jac_, typename CTraits::ParamJacType& actualJac_) : jac(jac_), actualJac(actualJac_), offs(0) {}
	template<typename T> void operator()(T) const
	{
		const int Index=T::value;
		typedef typename boost::fusion::result_of::at_c<typename CTraits::ParamType,Index>::type RVType;
		typedef typename boost::remove_reference<RVType>::type::ManifoldType MType; // WHY IS THIS REMOVE REF NEEDED???
		boost::fusion::at_c<Index>(actualJac)=jac.template block<CTraits::RetDim,MType::DeltaDim>(0,offs);
		offs+=MType::DeltaDim;
	}
	const Matrix<double,CTraits::RetDim,CTraits::TotalParamDeltaDim>& jac;
	typename CTraits::ParamJacType& actualJac;
	mutable int offs;
};

template<typename Constraint>
typename ConstraintTraits<Constraint>::ParamJacType computeJacobian(const Constraint& c, const typename ConstraintTraits<Constraint>::ParamVecType& params)
{
	typedef EigenFunctorAdapter<Constraint> EFA;
	EFA efa(c,params);
	NumericalDiff<decltype(efa),Central> ad(efa);
	typename EFA::JacobianType jac;
	typename EFA::InputType inp=EFA::InputType::Zero();
	ad.df(inp,jac);
	//std::cout << "JAC IS2 " << jac << std::endl;
	typedef ConstraintTraits<Constraint> CTraits;

	typename ConstraintTraits<Constraint>::ParamJacType actualJac;
	for_range<0,CTraits::NumParams>(DissectJacobian<Constraint>(jac,actualJac));
	return actualJac;
}

GSE_NS_END

#endif
