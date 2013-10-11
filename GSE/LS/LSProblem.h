#ifndef GSE_LS_LSPROBLEM_H
#define GSE_LS_LSPROBLEM_H

#define FUSION_MAX_SET_SIZE 20
#define FUSION_MAX_VECTOR_SIZE 20

#include "../GSEDefs.h"
#include "../ManifoldUtil.h"
#include "ConstraintTraits.h"
#include "NonlinearEstimator.h"
#include "RandomVariable.h"
#include "ConstraintRef.h"
#include "WrappedConstraint.h"
#include "../ThreadPool.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <boost/type_traits.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/fusion/sequence.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/container/set.hpp>
#include <boost/fusion/container/map.hpp>
#include <boost/fusion/container/generation/make_map.hpp>
#include <boost/fusion/container/generation/make_vector.hpp>
#include <boost/fusion/container/vector/convert.hpp>
#include <boost/fusion/include/make_map.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/include/is_sequence.hpp>
#include <boost/fusion/include/transform.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/at_key.hpp>
#include <boost/fusion/include/join.hpp>
#include <boost/fusion/include/sequence.hpp>
#include <boost/fusion/include/zip.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/filter_if.hpp>
#include <boost/function_types/result_type.hpp>
#include <boost/function_types/parameter_types.hpp>
#define BOOST_FUSION_INVOKE_MAX_ARITY 10
#include <boost/fusion/functional/invocation/invoke.hpp>
#include <boost/fusion/support/is_sequence.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/algorithm/query/find.hpp>
#include <boost/fusion/algorithm/transformation/transform.hpp>
#include <boost/fusion/algorithm/transformation/join.hpp>
#include <boost/fusion/sequence/intrinsic/at_key.hpp>
#include <boost/fusion/mpl.hpp>
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/mpl/print.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/range_c.hpp>
#include <stdexcept>
#include <unordered_map>

#include <fstream>

GSE_NS_BEGIN

using namespace Eigen;

//template <typename... Elems>
//struct RVHook;

//template <>
//struct RVHook<boost::fusion::set<> >
//{
//};

//template <typename Head, typename... Tail>
//struct RVHook<boost::fusion::set<Head, Tail...> > : public RVHook<boost::fusion::set<Tail...> >
//{
//	virtual void rvAdded(const Head& rv);
//};

//template <typename... Elems>
//struct ConstraintHook;

//template <>
//struct ConstraintHook<boost::fusion::set<> > {
//};

//template <typename Head, typename... Tail>
//struct ConstraintHook<boost::fusion::set<Head, Tail...> > : public ConstraintHook<boost::fusion::set<Tail...> >
//{
//	virtual void constraintAdded(const Head& h);
//};

//template<typename RVSet, typename ConstraintSet>
//struct ProblemHook : public RVHook<RVSet>, public ConstraintHook<ConstraintSet>
//{
//};



struct Offsets
{
	int stateOffs;
	int deltaOffs;
};

template <int first, int last, class Functor>
void for_range(Functor f)
{
	boost::fusion::for_each(typename boost::mpl::range_c<int,first,last>::type(),f);
}


#if 0 // Workaround for gcc bug
template <typename... Elems>
struct VarArgsToSet;

template <>
struct VarArgsToSet<> {
	typedef boost::fusion::set<> type;
}; // empty tuple

template <typename Head, typename... Tail>
struct VarArgsToSet<Head, Tail...> {
	typedef typename boost::fusion::result_of::push_back<typename VarArgsToSet<Tail...>::type,Head>::type type;
};


template<typename... Args>
class LSProblem
{
	typedef LSProblem<Args...> ThisType;

	// Convert Var Arg list to a set. The VarArgsToSet struct is only needed because gcc currently does not allow Passing Args... to make_set
	typedef typename boost::fusion::result_of::as_set<typename VarArgsToSet<Args...>::type>::type ArgSet;
#else
template<typename A1=boost::fusion::void_,typename A2=boost::fusion::void_,typename A3=boost::fusion::void_,typename A4=boost::fusion::void_,
		 typename A5=boost::fusion::void_,typename A6=boost::fusion::void_,typename A7=boost::fusion::void_,typename A8=boost::fusion::void_,
		 typename A9=boost::fusion::void_,typename A10=boost::fusion::void_,typename A11=boost::fusion::void_,typename A12=boost::fusion::void_,
                 typename A13=boost::fusion::void_,typename A14=boost::fusion::void_,typename A15=boost::fusion::void_>
class LSProblem
{
	typedef LSProblem<A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15> ThisType;
	//typedef typename boost::fusion::set<A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12> ArgSet;
	typedef typename boost::fusion::result_of::join<
	boost::fusion::set<A1,A2,A3,A4,A5,A6,A7,A8,A9,A10>,boost::fusion::set<A11,A12,A13,A14,A15> >::type ArgSet;
#endif

	// Filter those types which are Manifolds
	typedef typename boost::fusion::result_of::as_set<
	typename boost::fusion::result_of::filter_if<ArgSet,IsManifold<boost::mpl::_ > >::type
	>::type Types;
	// ... and those which are not -> They are assumed to be constraints
	typedef typename boost::fusion::result_of::as_set<
	typename boost::fusion::result_of::filter_if<ArgSet,IsNotManifold<boost::mpl::_ > >::type
	>::type ConstraintTypes;

	//-------------------------------------------------------------------------

	template<typename T>
	struct EmbedInRV
	{
		typedef RandomVariable<T> type;
	};
	typedef typename boost::mpl::transform<Types,EmbedInRV<boost::mpl::_> >::type RVTypes;

	template<typename T>
	struct EmbedInPair
	{
		typedef std::vector<RandomVariable<T> > type;//boost::fusion::pair<T,std::vector<RandomVariable<T> > > type;
	};
	//typedef typename boost::fusion::result_of::as_map<typename boost::mpl::transform<Types,EmbedInPair<boost::mpl::_> >::type>::type RVTypeVectors;
	typedef typename boost::mpl::transform<Types,EmbedInPair<boost::mpl::_> >::type RVTypeVectors;
	RVTypeVectors rvs;

	template<typename T>
	struct ToStateVec
	{
		typedef std::unordered_map<T,typename T::ManifoldType::Vec> type;
	};
	typedef typename boost::mpl::transform<RVTypes,ToStateVec<boost::mpl::_> >::type RVStateTypeVectors;
	RVStateTypeVectors rvStates;

	// test
	template<typename T>
	struct ToStateDeltaMat
	{
		typedef std::unordered_map<T,typename T::ManifoldType::DeltaMat> type;
	};
	typedef typename boost::mpl::transform<RVTypes,ToStateDeltaMat<boost::mpl::_> >::type RVStateDeltaMatVectors;
	RVStateDeltaMatVectors rvCovs;
	//

	template<typename T>
	struct EmbedInVector
	{
		typedef std::vector<T> type;
	};
	typedef typename boost::mpl::transform<ConstraintTypes,EmbedInVector<boost::mpl::_> >::type ConstraintTypeVectors;
	ConstraintTypeVectors constraints;

	int rvidCounter;

	template<typename T>
	struct ConvertToMap
	{
		typedef boost::fusion::pair<T,std::unordered_map<T,Offsets> > type;
	};
	typedef typename boost::fusion::result_of::as_map<
	typename boost::mpl::transform<RVTypes,ConvertToMap<boost::mpl::_> >::type
	>::type MapType;

	MapType rvToIndex;

	template<typename T>
	struct PairWithOffsets
	{
		typedef boost::fusion::pair<T,Offsets> type;
	};
	typedef typename boost::fusion::result_of::as_map<
	typename boost::mpl::transform<RVTypes,PairWithOffsets<boost::mpl::_> >::type
	>::type OffsetType;

	OffsetType offsets;
	int maxOffset;
	int maxDeltaOffset;
//////
public:
	//typedef ProblemHook<RVTypes,ConstraintTypes> HookType;




public:
	LSProblem() : rvidCounter{0}
	{
	}
	int getStateSize() const { return maxOffset; }
	int getDeltaSize() const { return maxDeltaOffset; }
	template<typename RVType>
	int getRVOffset(const RVType& rv) const
	{
		static_assert(boost::fusion::result_of::has_key<RVTypes,RVType>::type::value, "You did not add this RandomVariable type to the problem!");

		const std::unordered_map<RVType,Offsets>& actualRvToIndex=boost::fusion::at_key<RVType>(rvToIndex);
		auto it=actualRvToIndex.find(rv);
		if (it==actualRvToIndex.end())
			throw std::runtime_error("Invalid Random Variable!");
		return it->second.stateOffs;
	}
	template<typename RVType>
	int getRVDeltaOffset(const RVType& rv) const
	{
		static_assert(boost::fusion::result_of::has_key<RVTypes,RVType>::type::value, "You did not add this RandomVariable type to the problem!");

		const std::unordered_map<RVType,Offsets>& actualRvToIndex=boost::fusion::at_key<RVType>(rvToIndex);
		auto it=actualRvToIndex.find(rv);
		if (it==actualRvToIndex.end())
			throw std::runtime_error("Invalid Random Variable!");
		return it->second.deltaOffs;
	}

	const RVTypeVectors& getRVs() const { return rvs; }
	const ConstraintTypeVectors& getConstraints() const { return constraints; }

	void initializeSystem(LinearSystem& sys)
	{
		sys.mJTz.resize(getDeltaSize());

		// actually not needed for dense systems...
		generateNonZeroPattern(sys);
	}
	//-------------------------------------------------------------------------
	template<typename ConstraintT>
	ConstraintRef<ConstraintT> addConstraint(const ConstraintT& t)
	{
		static_assert(boost::fusion::result_of::has_key<ConstraintTypes,ConstraintT>::type::value, "You did not add this Constraint type to the problem!");

		std::vector<ConstraintT>& constraintList=boost::fusion::at_key<std::vector<ConstraintT> >(constraints);
		constraintList.push_back(t);

		static int consid=1;
		return {consid++};
	}
	//-------------------------------------------------------------------------
	template<typename ConstraintT>
	void popConstraint()
	{
		std::vector<ConstraintT>& constraintList=boost::fusion::at_key<std::vector<ConstraintT> >(constraints);
		constraintList.pop_back();
	}
	//-------------------------------------------------------------------------
	template<typename T>
	typename T::ManifoldType::Vec& state(T rv)
	{
		static_assert(boost::fusion::result_of::has_key<RVTypes,T>::type::value, "You did not add this RandomVariable type to the problem!");

		typedef std::unordered_map<T,typename T::ManifoldType::Vec> MapType;
		MapType& m=boost::fusion::at_key<MapType>(rvStates);
		auto it=m.find(rv);
		if (it==m.end())
			throw std::runtime_error("Invalid Random Variable!");
		return it->second;
	}
	//-------------------------------------------------------------------------
	template<typename T>
	const typename T::ManifoldType::Vec& state(T rv) const
	{
		static_assert(boost::fusion::result_of::has_key<RVTypes,T>::type::value, "You did not add this RandomVariable type to the problem!");

		typedef std::unordered_map<T,typename T::ManifoldType::Vec> MapType;
		const MapType& m=boost::fusion::at_key<MapType>(rvStates);
		auto it=m.find(rv);
		if (it==m.end())
			throw std::runtime_error("Invalid Random Variable!");
		return it->second;
	}
	//-------------------------------------------------------------------------
	template<typename T>
	const typename T::ManifoldType::DeltaMat& cov(T rv) const
	{
		static_assert(boost::fusion::result_of::has_key<RVTypes,T>::type::value, "You did not add this RandomVariable type to the problem!");

		typedef std::unordered_map<T,typename T::ManifoldType::DeltaMat> CovMapType;
		const CovMapType& m=boost::fusion::at_key<CovMapType>(rvCovs);
		auto it=m.find(rv);
		if (it==m.end())
			throw std::runtime_error("Invalid Random Variable!");
		return it->second;
	}
	//-------------------------------------------------------------------------
	template<typename T>
	typename T::ManifoldType::DeltaMat& cov(T rv)
	{
		static_assert(boost::fusion::result_of::has_key<RVTypes,T>::type::value, "You did not add this RandomVariable type to the problem!");

		typedef std::unordered_map<T,typename T::ManifoldType::DeltaMat> CovMapType;
		CovMapType& m=boost::fusion::at_key<CovMapType>(rvCovs);
		auto it=m.find(rv);
		if (it==m.end())
			throw std::runtime_error("Invalid Random Variable!");
		return it->second;
	}
	//-------------------------------------------------------------------------
	double optimize(bool verbose=true, bool computeCov=false)
	{
		NonlinearMinimizer<ThisType> nl{NonlinearMinimizer<ThisType>::TYPE::GAUSS_NEWTON};
		nl.setInitialGuess(getInitialGuess());
		nl.verbose=verbose;
		nl.work(*this,100);

		Eigen::SparseMatrix<double> xx;
		finish(nl.getResult(), xx);

		if (computeCov)
		{
			MatrixXd jtj(nl.sys.JTJ.rows(),nl.sys.JTJ.cols());
			for (int i=0;i<nl.sys.JTJ.rows();i++)
				for (int j=0;j<nl.sys.JTJ.cols();j++)
					jtj(i,j)=nl.sys.JTJ.coeff(i,j);
			std::cout << "jtj: " << jtj << std::endl;
			MatrixXd cov=nl.lastErr*jtj.inverse();// richtig
			//MatrixXd cov=jtj.inverse();//
			//std::cout << "LastErr is: " << nl.lastErr << std::endl;
			finishCov(cov);

		}
		return nl.lastErr;
	}
	//-------------------------------------------------------------------------
	double optimizeLM(bool verbose=true, bool computeCov=false)
	{
		NonlinearMinimizer<ThisType> nl{NonlinearMinimizer<ThisType>::TYPE::MARQUARDT};
		nl.setInitialGuess(getInitialGuess());
		nl.verbose=verbose;
		nl.work(*this,100);

		Eigen::SparseMatrix<double> xx;
		finish(nl.getResult(), xx);

		if (computeCov)
		{
			MatrixXd jtj(nl.sys.JTJ.rows(),nl.sys.JTJ.cols());
			for (int i=0;i<nl.sys.JTJ.rows();i++)
				for (int j=0;j<nl.sys.JTJ.cols();j++)
					jtj(i,j)=nl.sys.JTJ.coeff(i,j);
			std::cout << "jtj: " << jtj << std::endl;
			MatrixXd cov=nl.lastErr*jtj.inverse();// richtig
			//MatrixXd cov=jtj.inverse();//
			//std::cout << "LastErr is: " << nl.lastErr << std::endl;
			finishCov(cov);

		}
		return nl.lastErr;
	}
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
private:
	struct IndexAssigner
	{
		IndexAssigner(ThisType* pthis_) : pthis(pthis_), offs1(0), offs2(0) {}
		template<typename RVType> void operator()(const std::vector<RVType>& vec) const
		{
			boost::fusion::at_key<RVType>(pthis->offsets)=Offsets{offs1,offs2};

			constexpr int Dim=RVType::ManifoldType::Dim;
			constexpr int DeltaDim=RVType::ManifoldType::DeltaDim;

			std::unordered_map<RVType,Offsets>& actualRvToIndex=boost::fusion::at_key<RVType>(pthis->rvToIndex);
			for (int i=0;i<(int)vec.size();i++)
				actualRvToIndex[vec[i]]=Offsets{offs1+i*Dim,offs2+i*DeltaDim};

			offs1+=vec.size()*Dim;
			offs2+=vec.size()*DeltaDim;
		}
		ThisType* pthis;
		mutable int offs1;
		mutable int offs2;
	};
public:
	template<typename T>
	RandomVariable<T> addRandomVariable()
	{
		typedef RandomVariable<T> RVType;
		static_assert(boost::fusion::result_of::has_key<Types,T>::type::value, "You did not add this RandomVariable type to the problem!");

		std::vector<RVType>& rvList=boost::fusion::at_key<std::vector<RVType> >(rvs);
		RVType t{rvidCounter++};
		rvList.push_back(t);

		// insert initial rv state
		typedef std::unordered_map<RVType,typename RVType::ManifoldType::Vec> MapType;
		MapType& m=boost::fusion::at_key<MapType>(rvStates);
		m.insert({t,RVType::ManifoldType::Vec::Zero()});

		// insert initial rv state
		typedef std::unordered_map<RVType,typename RVType::ManifoldType::DeltaMat> CovMapType;
		CovMapType& cm=boost::fusion::at_key<CovMapType>(rvCovs);
		cm.insert({t,RVType::ManifoldType::DeltaMat::Identity()});

		// index update -- inefficient right now
		IndexAssigner ia{this};
		boost::fusion::for_each(rvs,ia);
		maxOffset=ia.offs1;
		maxDeltaOffset=ia.offs2;

		return t;
	}
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
private:
	template<typename T>
	struct ContainsRV
	{
		ContainsRV(RandomVariable<T> rv_) : rv(rv_),val(false) {}
		template<typename RVType> void operator()(const RVType& rv2) const
		{
			if (rv2.getId()==rv.getId())
				val=true;
		}
		RandomVariable<T> rv;
		mutable bool val;
	};
	template<typename T>
	struct RemoveConstraintsUsingRV
	{
		template<typename ConstraintType> void operator()(const std::vector<ConstraintType>& constrsc) const
		{
			// evil const cast
			std::vector<ConstraintType>& constrs=(std::vector<ConstraintType>&)constrsc;
			for (auto it=constrs.begin();it!=constrs.end();)
			{
				ContainsRV<T> crv{rv};
				boost::fusion::for_each(it->getParams(),crv);
				if (crv.val)
				{
					it=constrs.erase(it);
					//std::cout << "Removing Constraint!" << typeid(*it).name() << std::endl;
					continue;
				}
				++it;
			}
		}
		RandomVariable<T> rv;
	};
public:
	template<typename T>
	void removeRandomVariable(RandomVariable<T> rv)
	{
		static_assert(boost::fusion::result_of::has_key<Types,T>::type::value, "You did not add this RandomVariable type to the problem!");

		// This function is really inefficient right now
		// we should really have a map that tells us which constraints are dependent on a certain rv
		typedef RandomVariable<T> RVType;
		std::vector<RVType>& rvList=boost::fusion::at_key<std::vector<RVType> >(rvs);
		auto it=std::find(rvList.begin(),rvList.end(),rv);
		if (it==rvList.end())
			throw std::runtime_error("Invalid Random Variable!");
		rvList.erase(it);

		// now find all constraints which depend on this rv and delete those as well
		boost::fusion::for_each(constraints,RemoveConstraintsUsingRV<T>{rv});

		// remove initial guess
		typedef std::unordered_map<RVType,typename RVType::ManifoldType::Vec> MapType;
		MapType& m=boost::fusion::at_key<MapType>(rvStates);
		m.erase(m.find(rv));

		// index update -- inefficient right now
		IndexAssigner ia{this};
		boost::fusion::for_each(rvs,ia);
		maxOffset=ia.offs1;
		maxDeltaOffset=ia.offs2;
	}
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
private:
	struct InitialGuessExtractor
	{
		template<typename RVType> void operator()(const std::vector<RVType>& vec) const
		{
			//auto& actualPem=boost::fusion::at_key<T>();
			constexpr int Dim=RVType::ManifoldType::Dim;
			const int offs1=boost::fusion::at_key<RVType>(pthis->offsets).stateOffs;
			for (int i=0;i<(int)vec.size();i++)
				x.segment<Dim>(offs1+i*Dim)=pthis->state(vec[i]);
		}
		const ThisType* pthis;
		VectorXd& x;
	};
public:
	VectorXd getInitialGuess() const
	{
		VectorXd ret{getStateSize()};
		boost::fusion::for_each(rvs,InitialGuessExtractor{this,ret});
		return ret;
	}
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
private:
	struct DeltaAdder
	{
		template<typename RVType> void operator()(const std::vector<RVType>& vec) const
		{
			constexpr int Dim=RVType::ManifoldType::Dim;
			constexpr int DeltaDim=RVType::ManifoldType::DeltaDim;
			const int offs1=boost::fusion::at_key<RVType>(pthis->offsets).stateOffs;
			const int offs2=boost::fusion::at_key<RVType>(pthis->offsets).deltaOffs;
			for (int i=0;i<(int)vec.size();i++)
				x.segment<Dim>(offs1+i*Dim)=RVType::ManifoldType::manifoldAdd(x.segment<Dim>(offs1+i*Dim),delta.segment<DeltaDim>(offs2+i*DeltaDim));
		}
		const ThisType* pthis;
		VectorXd& x;
		const VectorXd& delta;
	};
public:
	void addDelta(VectorXd& x, const VectorXd& delta) const
	{
		////#include <boost/phoenix/scope/let.hpp>
		//#include <boost/phoenix/bind.hpp>
		//#include <boost/phoenix.hpp>
		//		using boost::phoenix::let;
		//		using boost::phoenix::ref;
		//		using boost::phoenix::cref;
		//		using boost::phoenix::local_names::_a;
		//		using boost::phoenix::local_names::_b;
		//		using boost::phoenix::local_names::_c;
		//		//using boost::phoenix::placeholder::_1;
		//		boost::fusion::for_each(graph.rvs,let(_a = ref(x), _b = cref(delta), _c = this)[
		//								int x;
		//								]);
		boost::fusion::for_each(rvs,DeltaAdder{this,x,delta});
	}
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
private:
	struct GetRVStateFromVector
	{
		//GetRVStateFromVector(ThisType* pthis_,const VectorXd& x_) : pthis(pthis_), x(x_) {}

		template <class T> struct result;
		template <class T> struct result<GetRVStateFromVector(const T&)>
		{
			typedef typename T::ManifoldType::Vec type;
		};

		template<typename T>
		typename ManifoldTraits<T>::Manifold::Vec operator()(RandomVariable<T> rv) const
		{
			int offs=pthis->getRVOffset(rv);
			constexpr int Dim=ManifoldTraits<T>::Manifold::Dim;
			return x.segment<Dim>(offs);
		}

		ThisType* pthis;
		const VectorXd& x;
	};
	template<typename CType, int Dim1>
	struct ForAllJacobians2
	{
		typedef ConstraintTraits<CType> CTraits;
		template<typename T> void operator()(T a) const
		{
			(void)a;
			//std::cout << "Iterating" << Dim1 << " " << T::value << std::endl;
			constexpr int Dim2=T::value;

			typedef typename boost::fusion::result_of::value_at_c<typename CTraits::ParamJacType,Dim1>::type JacA;
			typedef typename boost::fusion::result_of::value_at_c<typename CTraits::ParamJacType,Dim2>::type JacB;

			Matrix<double,JacA::ColsAtCompileTime,JacB::ColsAtCompileTime> jtj=boost::fusion::at_c<Dim1>(jac).transpose()*boost::fusion::at_c<Dim2>(jac);

			const int aOffset=pthis->getRVDeltaOffset(boost::fusion::at_c<Dim1>(params));
			const int bOffset=pthis->getRVDeltaOffset(boost::fusion::at_c<Dim2>(params));

			for (int i=0;i<JacA::ColsAtCompileTime;i++)
				for (int j=0;j<JacB::ColsAtCompileTime;j++)
					sys.JTJ.coeffRef(aOffset+i,bOffset+j)+=jtj(i,j);
		}
		ThisType* pthis;
		const typename CTraits::ParamType& params;
		typename CTraits::ParamJacType& jac;
		LinearSystem& sys;
	};
	template<typename CType>
	struct ForAllJacobians
	{
		typedef ConstraintTraits<CType> CTraits;
		template<typename T> void operator()(T a) const
		{
			(void)a;
			constexpr int Dim1=T::value;
			//std::cout << "Iterating" << T::value << std::endl;
			for_range<0,CTraits::NumParams>(ForAllJacobians2<CType,Dim1>{pthis,params,jac,sys});

			const int aRv=pthis->getRVDeltaOffset(boost::fusion::at_c<Dim1>(params));
			typedef typename boost::fusion::result_of::value_at_c<typename CTraits::ParamJacType,Dim1>::type JacA;

			sys.mJTz.segment<JacA::ColsAtCompileTime>(aRv)+=-boost::fusion::at_c<Dim1>(jac).transpose()*residual;

		}
		ThisType* pthis;
		const typename CTraits::ParamType& params;
		typename CTraits::RetType& residual;
		typename CTraits::ParamJacType& jac;
		LinearSystem& sys;
	};
	struct MakePointer
	{
		template <class T> struct result;
		template <class T> struct result<MakePointer(const T&)>
		{
			typedef T* type;
		};

		template<typename T>
		T* operator()(const T& t) const
		{
			return (T*)&t;
		}
	};
	struct UpdateSystem
	{
		template<typename CType> void operator()(const std::vector<CType>& constraints) const
		{
			//       Meas
			//    ----------------
			// Rvs|
			//    |
			//    |
			typedef ConstraintTraits<CType> CTraits;
			for (const CType& constraint : constraints)
			{
				// for every parameter: read the current value from the current state x
				typename CTraits::ParamVecType params=boost::fusion::transform(constraint.getParams(),GetRVStateFromVector{pthis,x});
				typename CTraits::ParamJacType jac;
#if 0
				typename CTraits::RetType residual;
				//it->error(params,residual,&jac);
#else
				//typedef typename boost::mpl::transform<typename CTraits::ParamJacType,MakePointer1<boost::mpl::_1> >::type JacPtrs;
				auto jacptr=boost::fusion::transform(jac,MakePointer{});

				auto fnParams=boost::fusion::push_front(boost::fusion::join(params,jacptr),&constraint);
				typename CTraits::RetType residual=boost::fusion::invoke(&CType::error,fnParams);
#endif
				for_range<0,CTraits::NumParams>(ForAllJacobians<CType>{pthis,constraint.getParams(),residual,jac,sys});
			}
		}
		ThisType* pthis;
		const VectorXd& x;
		LinearSystem& sys;
	};
public:
	void updateSystem(const VectorXd& x, LinearSystem& sys)
	{
		updateSystem_par(x,sys);
		return;
		sys.mJTz=VectorXd::Zero(getDeltaSize());

		// set all existing "non-zero" entries to 0
		for (int k=0; k<sys.JTJ.outerSize(); ++k)
		{
			for (Eigen::SparseMatrix<double>::InnerIterator it{sys.JTJ,k}; it; ++it)
			{
				it.valueRef()=0;
			}
		}

		boost::fusion::for_each(getConstraints(),UpdateSystem{this,x,sys});

		//for (int i=0;i<6;i++)
		//sys.JTJ.coeffRef(i,i)+=1;
		//sys.JTJ.coeffRef(0,0)+=1;
		//sys.JTJ.coeffRef(1,1)+=1;
		//sys.JTJ.coeffRef(2,2)+=1;
		//std::cout << sys.JTJ << std::endl;
	}
private:
	struct UpdateSystemPar
	{
		template<typename CType> void operator()(const std::vector<CType>& constraints) const
		{
			//       Meas
			//    ----------------
			// Rvs|
			//    |
			//    |
			typedef ConstraintTraits<CType> CTraits;
			int task=0;
			for (const CType& constraint : constraints)
			{
				int th=task%numThreads;
				pools[th]->pushTask([this,th,&constraint]()
				{
					// for every parameter: read the current value from the current state x
					typename CTraits::ParamVecType params=boost::fusion::transform(constraint.getParams(),GetRVStateFromVector{pthis,x});
					typename CTraits::ParamJacType jac;
#if 0
					typename CTraits::RetType residual;
					//it->error(params,residual,&jac);
#else
					//typedef typename boost::mpl::transform<typename CTraits::ParamJacType,MakePointer1<boost::mpl::_1> >::type JacPtrs;
					auto jacptr=boost::fusion::transform(jac,MakePointer{});

					auto fnParams=boost::fusion::push_front(boost::fusion::join(params,jacptr),&constraint);
					typename CTraits::RetType residual=boost::fusion::invoke(&CType::error,fnParams);
#endif
					for_range<0,CTraits::NumParams>(ForAllJacobians<CType>{pthis,constraint.getParams(),residual,jac,systems[th]});
				});
				task++;
			}
		}
		ThisType* pthis;
		const VectorXd& x;
		LinearSystem* systems;
		ThreadPool** pools;
		int numThreads;
	};
public:
	void updateSystem_par(const VectorXd& x, LinearSystem& sys)
	{
		sys.mJTz=VectorXd::Zero(getDeltaSize());

		// set all existing "non-zero" entries to 0
		for (int k=0; k<sys.JTJ.outerSize(); ++k)
		{
			for (Eigen::SparseMatrix<double>::InnerIterator it{sys.JTJ,k}; it; ++it)
			{
				it.valueRef()=0;
			}
		}

		const int numThreads=4;
		ThreadPool* pools[numThreads];
		LinearSystem systems[numThreads];
		for (int i=0;i<numThreads;i++)
		{
			pools[i]=new ThreadPool(1);
			systems[i]=sys;
		}

		boost::fusion::for_each(getConstraints(),UpdateSystemPar{this,x,&systems[0],&pools[0],numThreads});

		for (int i=0;i<numThreads;i++)
		{
			pools[i]->joinAllTasks();
			delete pools[i];
			sys.JTJ+=systems[i].JTJ;
			sys.mJTz+=systems[i].mJTz;
		}
	}
	struct UpdateSystemSC
	{
		template<typename CType> void operator()(const std::vector<CType>& constraints) const
		{
			//       Meas
			//    ----------------
			// Rvs|
			//    |
			//    |
			typedef ConstraintTraits<CType> CTraits;
			//for (int i=0;i<(int)constraints.size();i++)
			if (constraints.empty())
				return;
			int i=rand()%constraints.size();
			{
				const CType& constraint=constraints[i];
				// for every parameter: read the current value from the current state x
				typename CTraits::ParamVecType params=boost::fusion::transform(constraint.getParams(),GetRVStateFromVector{pthis,x});
				typename CTraits::ParamJacType jac;
#if 0
				typename CTraits::RetType residual;
				//it->error(params,residual,&jac);
#else
				//typedef typename boost::mpl::transform<typename CTraits::ParamJacType,MakePointer1<boost::mpl::_1> >::type JacPtrs;
				auto jacptr=boost::fusion::transform(jac,MakePointer());

				auto fnParams=boost::fusion::push_front(boost::fusion::join(params,jacptr),&constraint);
				typename CTraits::RetType residual=boost::fusion::invoke(&CType::error,fnParams);
#endif
				for_range<0,CTraits::NumParams>(ForAllJacobians<CType>{pthis,constraint.getParams(),residual,jac,sys});
			}
		}
		ThisType* pthis;
		const VectorXd& x;
		LinearSystem& sys;
	};
	void updateSystemSingleConstraint(const VectorXd& x, LinearSystem& sys)
	{
		sys.mJTz=VectorXd::Zero(getDeltaSize());

		boost::fusion::for_each(getConstraints(),UpdateSystemSC{this,x,sys});

	}
	void updateDenseSystem(const VectorXd& x, LinearSystem& sys)
	{
		(void)x;
		(void)sys;
		assert(0 && "Not implemented");
	}
	//-----------------------------------------------------------------------------
	//-----------------------------------------------------------------------------
private:
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
	struct ComputeError
	{
		ComputeError(ThisType* pthis_, const VectorXd& x_, VectorXd& err_) : pthis(pthis_), x(x_), err(err_), measOffs(0) {}
		template<typename CType> void operator()(const std::vector<CType>& constraints) const
		{
			for (const CType& constraint : constraints)
			{
				typedef ConstraintTraits<CType> CTraits;
				typename CTraits::ParamVecType params=boost::fusion::transform(constraint.getParams(),GetRVStateFromVector{pthis,x});
#if 0
				typename CTraits::RetType residual;
				it->error(params,residual,nullptr);
#else
				typename CTraits::ParamJacType jac;
				auto jacptr=boost::fusion::transform(jac,MakeNullPointer{});

				//boost::mpl::print<decltype(jacptr)> dsdsads;
				auto fnParams=boost::fusion::push_front(boost::fusion::join(params,jacptr),&constraint);
				//boost::mpl::print<decltype(fnParams)> asdiohsadisaod;
				typename CTraits::RetType residual=boost::fusion::invoke(&CType::error,fnParams);
#endif

				constexpr int DeltaDim=ManifoldTraits<typename CTraits::RetType>::Manifold::DeltaDim;

				err.segment<DeltaDim>(measOffs)=residual;
				measOffs+=DeltaDim;
			}

		}
		ThisType* pthis;
		const VectorXd& x;
		VectorXd& err;
		mutable int measOffs;
	};
public:
	VectorXd computeError(const VectorXd& x)
	{
		MeasDim md;
		boost::fusion::for_each(getConstraints(),md);
		const int measurementDim=md.measDim;

		VectorXd err(measurementDim);
		boost::fusion::for_each(getConstraints(),ComputeError{this,x,err});
		return err;
	}
	//-----------------------------------------------------------------------------
	//-----------------------------------------------------------------------------
private:
	struct MeasDim
	{
		MeasDim() : measDim(0) {}
		template<typename T> void operator()(const std::vector<T>& vec) const
		{
			measDim+=vec.size()*ConstraintTraits<T>::RetType::RowsAtCompileTime;
		}
		mutable int measDim;
	};
	struct InsertOnesInner
	{
		template<typename T> void operator()(RandomVariable<T> paramRv) const
		{
			const int rvOffsFrom=pthis->getRVDeltaOffset(paramRv);

			for (int j=0;j<measDim;j++)
				for (int i=0;i<RandomVariable<T>::ManifoldType::DeltaDim;i++)
					//triplets.push_back(Eigen::Triplet<double>{rvOffsFrom+i,measOffset+j,1});
                    triplets.push_back({(unsigned int)(rvOffsFrom+i),(unsigned int)(measOffset+j),1});
		}
		ThisType* pthis;
		std::vector<Eigen::Triplet<double> >& triplets;
		mutable int measOffset;
		mutable int measDim;
	};
	struct InsertOnes
	{
		InsertOnes(ThisType* pthis_, std::vector<Eigen::Triplet<double> >& triplets_) : pthis(pthis_), triplets(triplets_), measOffset(0) {}
		template<typename CType> void operator()(const std::vector<CType>& constraints) const
		{
			typedef ConstraintTraits<CType> CTraits;
			constexpr int DeltaDim=CTraits::RetType::RowsAtCompileTime;
			for (const CType& constraint : constraints)
			{
				// loop over all params of this measurement
				boost::fusion::for_each(constraint.getParams(),InsertOnesInner{pthis,triplets, measOffset, DeltaDim});

				measOffset+=DeltaDim;
			}
		}
		ThisType* pthis;
		std::vector<Eigen::Triplet<double> >& triplets;
		mutable int measOffset;
	};
	void generateNonZeroPattern(LinearSystem& sys)
	{
		MeasDim md;
		boost::fusion::for_each(getConstraints(),md);
		const int measurementDim=md.measDim;

		std::vector<Eigen::Triplet<double> > triplets;
		boost::fusion::for_each(getConstraints(),InsertOnes{this,triplets});

		Eigen::SparseMatrix<double> HT{getDeltaSize(),measurementDim};
		HT.setFromTriplets(triplets.begin(),triplets.end());

		HT.finalize();
		sys.JTJ=HT*HT.transpose();

		//std::cout << HT << std::endl;
	}
	//-------------------------------------------------------------------------
	struct Finisher
	{
		template<typename RVType> void operator()(const std::vector<RVType>& vec) const
		{
			constexpr int Dim=RVType::ManifoldType::Dim;
			constexpr int DeltaDim=RVType::ManifoldType::DeltaDim;
			int offs1=boost::fusion::at_key<RVType>(pthis->offsets).stateOffs;
			int offs2=boost::fusion::at_key<RVType>(pthis->offsets).deltaOffs;
			for (int i=0;i<(int)vec.size();i++)
			{
				pthis->state(vec[i])=x.segment<Dim>(offs1);
				offs1+=Dim;
				offs2+=DeltaDim;
			}
		}
		ThisType* pthis;
		const VectorXd& x;
	};
	void finish(const VectorXd& x, /*const*/ Eigen::SparseMatrix<double>& cov)
	{
		boost::fusion::for_each(rvs,Finisher{this,x});
		(void)cov; // todo: use cov
	}
	void finishDense(const VectorXd& x, /*const*/ MatrixXd& cov)
	{
		boost::fusion::for_each(rvs,Finisher{this,x});
		(void)cov; // todo: use cov
	}
	//-------------------------------------------------------------------------
	struct CovFinisher
	{
		template<typename RVType> void operator()(const std::vector<RVType>& vec) const
		{
			constexpr int DeltaDim=RVType::ManifoldType::DeltaDim;
			int offs2=boost::fusion::at_key<RVType>(pthis->offsets).deltaOffs;
			for (int i=0;i<(int)vec.size();i++)
			{
				pthis->cov(vec[i])=cov.block<DeltaDim,DeltaDim>(offs2,offs2);
				offs2+=DeltaDim;
			}
		}
		ThisType* pthis;
		const MatrixXd& cov;
	};
	void finishCov(const Eigen::MatrixXd& cov)
	{
		boost::fusion::for_each(rvs,CovFinisher{this,cov});
		(void)cov; // todo: use cov
	}

	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
	template<typename F>
	struct RVIterator
	{
		template<typename RVType> void operator()(const std::vector<RVType>& vec) const
		{
			for (auto rv : vec)
				f(rv);
		}
		F& f;
	};
public:
	template<typename F>
	void forEachRV(F&& f) const
	{
		boost::fusion::for_each(rvs,RVIterator<F>{f});
	}
	template<typename RVType, typename F>
	void forEachRV(F&& f) const
	{
		auto& rvList=boost::fusion::at_key<std::vector<RandomVariable<RVType> > >(rvs);
		for (auto rv : rvList)
			f(rv);
	}
private:
	template<typename F>
	struct ConstraintIterator
	{
		template<typename CType> void operator()(const std::vector<CType>& constraints) const
		{
			for (auto c : constraints)
				f(c);
		}
		F& f;
	};
public:
	template<typename F>
	void forEachConstraint(F&& f) const
	{
		boost::fusion::for_each(constraints,ConstraintIterator<F>{f});
	}
	template<typename CType, typename F>
	void forEachConstraint(F&& f) const
	{
		const std::vector<CType>& constraintList=boost::fusion::at_key<std::vector<CType> >(constraints);
		for (auto c : constraintList)
			f(c);
	}
private:


	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------

	struct RVWriter
	{
		template<typename RVType> void operator()(const std::vector<RVType>& vec) const
		{
			for (auto& rv : vec)
			{
				out << rv.getId() << "[label=\"" << typeid(RVType).name() << "_" << rv.getId() << "\"] ";
				//out << rv.getId() << "[label=\"" << rv.getId() << "\"] ";
			}
		}
		std::ofstream& out;
	};
	struct ConstraintEdgeWriter
	{
		template<typename T> void operator()(RandomVariable<T> paramRv) const
		{
			out << paramRv.getId() << " -- _" << cname << "_" << cid << "\n";
		}
		std::ofstream& out;
		std::string cname;
		int cid;
	};
	struct ConstraintWriter
	{
		template<typename CType> void operator()(const std::vector<CType>& constraints) const
		{
			int cid=0;
			for (auto& c : constraints)
			{
				std::string cname=typeid(CType).name();
				out << "node [shape=box]; _" << cname << "_" << cid << "[label=\"" << cname <<  "\"]\n";
				//out << "node [shape=point]; _" << cname << "_" << cid << "[label=\"" << cname <<  "\"]\n";
				boost::fusion::for_each(c.getParams(),ConstraintEdgeWriter{out,cname,cid});
				out << "\n";
				cid++;
			}
		}
		std::ofstream& out;
	};
public:
	void writeDot(std::ofstream& out)
	{
		out << "graph FG {\n";
		out << "   nodesep=0.6;\n";
		out << "   sep=\"+25,25\";\n";
		out << "   overlap=scalexy;\n\n";

//		graph FG {
//			node [shape=ellipse] A[label="bla"] B C

//			node [shape=box]; C1[label="honk"]
//			A -- C1;
//			B -- C1;

//			node [shape=box]; C2;
//			C -- C2;
//			B -- C2;
//		}
		out << "node [shape=ellipse] ";
		boost::fusion::for_each(rvs,RVWriter{out});
		out << "\n\n";
		boost::fusion::for_each(constraints,ConstraintWriter{out});
		out << "}" << std::endl;
	}

};



//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

GSE_NS_END


#endif
