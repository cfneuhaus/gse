#ifndef GSE_LS_CONSTRAINTTRAITS_H
#define GSE_LS_CONSTRAINTTRAITS_H

#include "../GSEDefs.h"
#include "RandomVariable.h"
#include <boost/fusion/container/generation/make_vector.hpp>
#include <boost/function_types/result_type.hpp>
#include <boost/function_types/parameter_types.hpp>
#include <boost/fusion/mpl.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/join.hpp>
#include <boost/fusion/include/zip.hpp>


#include <boost/mpl/accumulate.hpp>

GSE_NS_BEGIN

template<typename T>
struct IsEigenVector : public boost::mpl::false_
{
};
template<int N>
struct IsEigenVector<Matrix<double,N,1> > : public boost::mpl::true_
{
};


#if 1

#define _remRef(F) typename boost::remove_reference<F>::type
#define _remConst(F) typename boost::remove_const<F>::type
#define _makeVec(F) typename boost::fusion::result_of::make_vector<F>::type
#define _if(C,T,E) typename boost::mpl::if_c<C,T,E>::type
#define _declare(A,B) typedef B A
#define _isSequence(X) boost::fusion::traits::is_sequence<X>::value
#define _asVector(X) typename boost::fusion::result_of::as_vector<X>::type
#define _popFront(X) typename boost::fusion::result_of::pop_front<X>::type
#define _join(X,Y) typename boost::fusion::result_of::join<X,Y>::type
#define _transform(X,Y) typename boost::mpl::transform<X,Y>::type
#define _zip(X,Y) typename boost::fusion::result_of::zip<X,Y>::type
#define _resultType(X) typename boost::function_types::result_type<X>::type
#define _paramTypes(X) typename boost::function_types::parameter_types<X>::type
#define _size(X) boost::fusion::result_of::size<X>::value
#define _call(X,Y) typename X<Y>::type
#define _return(...) typedef __VA_ARGS__ type

#define _fx(N,X,Y) template<typename X> \
	struct N \
{ \
	typedef Y type; \
}

#define _fxbegin(N,X) template<typename X> \
	struct N

#define _returnInt(X)	enum { value=X }; \
	typedef boost::mpl::integral_c_tag tag; \
	typedef int value_type \


template<typename T>
struct ConstraintTraits
{
	_declare(ParamType1,_remRef(
				 _remConst(
					 _resultType(decltype(&T::getParams))
					 )
				 )
			 );

	_declare(ParamType,_if(_isSequence(ParamType1),
						   ParamType1,
						   _makeVec(ParamType1)));

	static_assert(_isSequence(ParamType), "getParams() must return a boost::fusion::vector of RandomVariables!");

	enum { NumParams=_size(ParamType) };

	//template<typename R,int N>
	//struct GetDim
	//{
	//enum { value=N+R::Manifold::Dim };
	//};
	//boost::fusion
	//enum { ParamDim=boost::fusion::result_of::fold<ParamType,0,GetDim<boost::mpl::_1,boost::mpl::_2> >::value; }
	/*struct plus_mpl
		{
			template <class T1, class T2>
			struct apply
			{
			   typedef typename boost::mpl::plus<T1,T2::Manifold::Dim>::type type;
			};
		};
	enum { ParamDim=boost::mpl::accumulate<ParamType, boost::mpl::int_<0>, plus_mpl >::type::value }*/

	// Compute total parameter dimension
	_fxbegin(GetDim,R)
	{
		_returnInt(R::ManifoldType::Dim);
	};

	enum { TotalParamDim=boost::mpl::fold<ParamType, boost::mpl::int_<0>, boost::mpl::plus< GetDim<boost::mpl::_2>,boost::mpl::_1>  >::type::value };
	_fxbegin(GetDeltaDim,R)
	{
		_returnInt(R::ManifoldType::DeltaDim);
	};
	enum { TotalParamDeltaDim=boost::mpl::fold<ParamType, boost::mpl::int_<0>, boost::mpl::plus< GetDeltaDim<boost::mpl::_2>,boost::mpl::_1>  >::type::value };



	/////////////////////
	//template<typename X>
	//struct ConvertToVecType
	//{
	//static_assert(IsRandomVariable<X>::value, "getParams() must return a boost::fusion::vector of RandomVariables!");

	//typedef typename X::ManifoldType::Vec type;
	//};
	_fxbegin(ConvertToVecType,X)
	{
		_return(typename X::ManifoldType::Vec);
	};
	_declare(ParamVecType,_transform(ParamType,ConvertToVecType<boost::mpl::_>));

	/////////////////////
	_fx(ConvertToDeltaVecType,X,typename X::ManifoldType::DeltaVec);
	_declare(ParamDeltaVecType,_transform(ParamType,ConvertToDeltaVecType<boost::mpl::_>));

	/////////////////////

	_declare(RetParam,_resultType(decltype(&T::error)));
	_declare(RetType,_remRef(RetParam));

	static_assert(IsEigenVector<RetType>::value, "Constraint error function must return an Eigen row-vector!");

	static constexpr int RetDim=RetType::RowsAtCompileTime;
	/////////////////////
	//_fx(ConvertToJacType,X,(Matrix<double,RetDim,X::ManifoldType::DeltaDim>));
	_fxbegin(ConvertToJacType,X)
	{
		_return(Matrix<double,RetDim,X::ManifoldType::DeltaDim>);
	};

	_declare(ParamJacType,_transform(ParamType,ConvertToJacType<boost::mpl::_>));

	/////////////////////
	// Check Parameters of error function

	//_fx(ConvertToJacPtrType,X,(Matrix<double,RetDim,X::ManifoldType::DeltaDim>*));
	_fxbegin(ConvertToJacPtrType,X)
	{
		_return(Matrix<double,RetDim,X::ManifoldType::DeltaDim>*);
	};
	_declare(ParamJacPtrType,_transform(ParamType,ConvertToJacPtrType<boost::mpl::_>));

	_fx(EmbedInConstRef,X,const X&);
	_declare(ExpectedParams,_asVector(
				 _join(_transform(ParamVecType,EmbedInConstRef<boost::mpl::_1>),
					   ParamJacPtrType)));

	_declare(ActualParams,_asVector(
				 _popFront(
					 _paramTypes(decltype(&T::error))
					 )
				 )
			 );

	//boost::mpl::print<ActualParams> sdsd;

	//static_assert(boost::is_same<ExpectedParams,ActualParams>::value, "The parameters to the constraints error function are incorrect!");
	//static_assert(boost::is_convertible<ExpectedParams,ActualParams>::value, "The parameters to the constraints error function are incorrect!");

	//typedef typename boost::fusion::result_of::as_vector<typename boost::fusion::result_of::zip<ActualParams,ExpectedParams>::type>::type ZippedParams;
	_declare(ZippedParams,_asVector(_zip(ActualParams,ExpectedParams)));

	//boost::mpl::print<ZippedParams> sddsdsd;

	template<typename X>
	struct CheckPair
	{
		static_assert(boost::is_convertible<typename boost::fusion::result_of::at_c<X,0>::type,
					  typename boost::fusion::result_of::at_c<X,1>::type >::value,"The parameters to the constraint's error function are incorrect! Note that the Jacobian has a dimension of NxM where N=dimension of return vector and M=dimension of input random variable (manifold deltadim)");

		//boost::mpl::print<typename boost::fusion::result_of::at_c<X,0>::type> sddsdsd;
	};
	//typedef typename boost::mpl::transform<ZippedParams,CheckPair<boost::mpl::_> >::type CheckParamType;
	_declare(CheckParamType,_transform(ZippedParams,CheckPair<boost::mpl::_>));

};


#else
template<typename T>
struct ConstraintTraits
{
	typedef typename boost::remove_reference<
	typename boost::remove_const<
	typename boost::function_types::result_type<decltype(&T::getParams)>::type
	>::type
	>::type ParamType1;

	typedef typename boost::mpl::if_c<
	boost::fusion::traits::is_sequence<ParamType1>::value,
	ParamType1,
	typename boost::fusion::result_of::make_vector<ParamType1>::type>::type ParamType; // hrmpf

	static_assert(boost::fusion::traits::is_sequence<ParamType>::value, "getParams() must return a boost::fusion::vector of RandomVariables!");

	enum { NumParams=boost::fusion::result_of::size<ParamType>::value };

	//template<typename R,int N>
	//struct GetDim
	//{
	//enum { value=N+R::Manifold::Dim };
	//};
	//boost::fusion
	//enum { ParamDim=boost::fusion::result_of::fold<ParamType,0,GetDim<boost::mpl::_1,boost::mpl::_2> >::value; }
	/*struct plus_mpl
		{
			template <class T1, class T2>
			struct apply
			{
			   typedef typename boost::mpl::plus<T1,T2::Manifold::Dim>::type type;
			};
		};
	enum { ParamDim=boost::mpl::accumulate<ParamType, boost::mpl::int_<0>, plus_mpl >::type::value }*/

	// Compute total parameter dimension
	template<typename R>
	struct GetDim
	{
		enum { value=R::ManifoldType::Dim };
		typedef boost::mpl::integral_c_tag tag;
		typedef int value_type;
	};
	enum { TotalParamDim=boost::mpl::fold<ParamType, boost::mpl::int_<0>, boost::mpl::plus< GetDim<boost::mpl::_2>,boost::mpl::_1>  >::type::value };
	template<typename R>
	struct GetDeltaDim
	{
		enum { value=R::ManifoldType::DeltaDim };
		typedef boost::mpl::integral_c_tag tag;
		typedef int value_type;
	};
	enum { TotalParamDeltaDim=boost::mpl::fold<ParamType, boost::mpl::int_<0>, boost::mpl::plus< GetDeltaDim<boost::mpl::_2>,boost::mpl::_1>  >::type::value };



	/////////////////////
	template<typename X>
	struct ConvertToVecType
	{
		static_assert(IsRandomVariable<X>::value, "getParams() must return a boost::fusion::vector of RandomVariables!");

		typedef typename X::ManifoldType::Vec type;
	};
	typedef typename boost::mpl::transform<ParamType,ConvertToVecType<boost::mpl::_> >::type ParamVecType;

	/////////////////////
	template<typename X>
	struct ConvertToDeltaVecType
	{
		typedef typename X::ManifoldType::DeltaVec type;
	};
	typedef typename boost::mpl::transform<ParamType,ConvertToDeltaVecType<boost::mpl::_> >::type ParamDeltaVecType;

	/////////////////////

	typedef typename boost::function_types::result_type<decltype(&T::error)>::type RetParam;
	typedef typename boost::remove_reference<RetParam>::type RetType;

	static_assert(IsEigenVector<RetType>::value, "Constraint error function must return an Eigen row-vector!");

	static constexpr int RetDim=RetType::RowsAtCompileTime;
	/////////////////////
	template<typename X>
	struct ConvertToJacType
	{
		typedef Matrix<double,RetDim,X::ManifoldType::DeltaDim> type;
	};
	typedef typename boost::mpl::transform<ParamType,ConvertToJacType<boost::mpl::_> >::type ParamJacType;

	/////////////////////
	// Check Parameters of error function

	template<typename X>
	struct ConvertToJacPtrType
	{
		typedef Matrix<double,RetDim,X::ManifoldType::DeltaDim>* type;
	};
	typedef typename boost::mpl::transform<ParamType,ConvertToJacPtrType<boost::mpl::_> >::type ParamJacPtrType;

	template<typename X>
	struct EmbedInConstRef
	{
		typedef const X& type;
	};

	typedef typename boost::fusion::result_of::as_vector<
	typename boost::fusion::result_of::join<
	typename boost::mpl::transform<
	ParamVecType,EmbedInConstRef<boost::mpl::_1>
	>::type
	,ParamJacPtrType>
	::type>::type ExpectedParams;

	typedef typename boost::fusion::result_of::as_vector<
	typename boost::fusion::result_of::pop_front<
	typename boost::function_types::parameter_types<decltype(&T::error)>::type
	>::type > ::type ActualParams;


	//boost::mpl::print<ActualParams> sdsd;

	//static_assert(boost::is_same<ExpectedParams,ActualParams>::value, "The parameters to the constraints error function are incorrect!");
	//static_assert(boost::is_convertible<ExpectedParams,ActualParams>::value, "The parameters to the constraints error function are incorrect!");

	typedef typename boost::fusion::result_of::as_vector<typename boost::fusion::result_of::zip<ActualParams,ExpectedParams>::type>::type ZippedParams;

	//boost::mpl::print<ZippedParams> sddsdsd;

	template<typename X>
	struct CheckPair
	{
		static_assert(boost::is_convertible<typename boost::fusion::result_of::at_c<X,0>::type,
					  typename boost::fusion::result_of::at_c<X,1>::type >::value,"The parameters to the constraint's error function are incorrect! Note that the Jacobian has a dimension of NxM where N=dimension of return vector and M=dimension of input random variable (manifold deltadim)");

		//boost::mpl::print<typename boost::fusion::result_of::at_c<X,0>::type> sddsdsd;
	};
	typedef typename boost::mpl::transform<ZippedParams,CheckPair<boost::mpl::_> >::type CheckParamType;

};
#endif

GSE_NS_END

#endif
