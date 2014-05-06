#if __cplusplus==201103L
#include "LSProblem11.h"
#else

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
#include <boost/mpl/for_each.hpp>
#include <stdexcept>
#include <unordered_map>

#include <fstream>

GSE_NS_BEGIN

using namespace Eigen;

struct Offsets
{
	int stateOffs;
	int deltaOffs;
};

template<typename... Args>
class LSProblem
{
	typedef LSProblem<Args...> ThisType;

	// Convert Var Arg list to a set. The VarArgsToSet struct is only needed because gcc currently does not allow Passing Args... to make_set
	typedef boost::fusion::set<Args...> ArgSet;

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
	auto getRVSegment(Eigen::VectorXd& x, const RVType& rv)
	{
		int offs=this->getRVOffset(rv);
		constexpr int Dim=RVType::ManifoldType::Dim;
		return x.segment<Dim>(offs);
	}
	template<typename RVType>
	auto getRVSegment(const Eigen::VectorXd& x, const RVType& rv)
	{
		int offs=this->getRVOffset(rv);
		constexpr int Dim=RVType::ManifoldType::Dim;
		return x.segment<Dim>(offs);
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
		maxOffset=0;
		maxDeltaOffset=0;
		boost::fusion::for_each(rvs,[&]<typename RVType>(const std::vector<RVType>& vec)
		{
			boost::fusion::at_key<RVType>(offsets)=Offsets{maxOffset,maxDeltaOffset};

			constexpr int Dim=RVType::ManifoldType::Dim;
			constexpr int DeltaDim=RVType::ManifoldType::DeltaDim;

			std::unordered_map<RVType,Offsets>& actualRvToIndex=boost::fusion::at_key<RVType>(rvToIndex);
			for (int i=0;i<(int)vec.size();i++)
				actualRvToIndex[vec[i]]=Offsets{maxOffset+i*Dim,maxDeltaOffset+i*DeltaDim};

			maxOffset+=vec.size()*Dim;
			maxDeltaOffset+=vec.size()*DeltaDim;

		});

		return t;
	}
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
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
		boost::fusion::for_each(constraints,[&]<typename ConstraintType>(const std::vector<ConstraintType>& constrsc)
		{
			// evil const cast
			std::vector<ConstraintType>& constrs=(std::vector<ConstraintType>&)constrsc;
			for (auto it=constrs.begin();it!=constrs.end();)
			{
				bool val=false;
				boost::fusion::for_each(it->getParams(),[&]<typename RVType>(const RVType& rv2)
				{
					if (rv2.getId()==rv.getId())
						val=true;
				});
				if (val)
				{
					it=constrs.erase(it);
					//std::cout << "Removing Constraint!" << typeid(*it).name() << std::endl;
					continue;
				}
				++it;
			}
		});

		// remove initial guess
		typedef std::unordered_map<RVType,typename RVType::ManifoldType::Vec> MapType;
		MapType& m=boost::fusion::at_key<MapType>(rvStates);
		m.erase(m.find(rv));

		// index update -- inefficient right now
		maxOffset=0;
		maxDeltaOffset=0;
		boost::fusion::for_each(rvs,[&]<typename RVType>(const std::vector<RVType>& vec)
		{
			boost::fusion::at_key<RVType>(offsets)=Offsets{maxOffset,maxDeltaOffset};

			constexpr int Dim=RVType::ManifoldType::Dim;
			constexpr int DeltaDim=RVType::ManifoldType::DeltaDim;

			std::unordered_map<RVType,Offsets>& actualRvToIndex=boost::fusion::at_key<RVType>(rvToIndex);
			for (int i=0;i<(int)vec.size();i++)
				actualRvToIndex[vec[i]]=Offsets{maxOffset+i*Dim,maxDeltaOffset+i*DeltaDim};

			maxOffset+=vec.size()*Dim;
			maxDeltaOffset+=vec.size()*DeltaDim;

		});

	}
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
	VectorXd getInitialGuess() const
	{
		VectorXd ret{getStateSize()};
		boost::fusion::for_each(rvs,[&]<typename RVType>(const std::vector<RVType>& vec)
		{
			//auto& actualPem=boost::fusion::at_key<T>();
			constexpr int Dim=RVType::ManifoldType::Dim;
			const int offs1=boost::fusion::at_key<RVType>(offsets).stateOffs;
			for (int i=0;i<(int)vec.size();i++)
				ret.segment<Dim>(offs1+i*Dim)=this->state(vec[i]);
		});
		return ret;
	}
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
	void addDelta(VectorXd& x, const VectorXd& delta) const
	{
		boost::fusion::for_each(rvs,[&]<typename RVType>(const std::vector<RVType>& vec)
		{
			constexpr int Dim=RVType::ManifoldType::Dim;
			constexpr int DeltaDim=RVType::ManifoldType::DeltaDim;
			const int offs1=boost::fusion::at_key<RVType>(this->offsets).stateOffs;
			const int offs2=boost::fusion::at_key<RVType>(this->offsets).deltaOffs;
			for (int i=0;i<(int)vec.size();i++)
				x.segment<Dim>(offs1+i*Dim)=RVType::ManifoldType::manifoldAdd(x.segment<Dim>(offs1+i*Dim),delta.segment<DeltaDim>(offs2+i*DeltaDim));
		});
	}
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
private:
	template<typename CType>
	void addConstraintToSystem(const Eigen::VectorXd& x, const CType& constraint, LinearSystem& sys)
	{
		typedef ConstraintTraits<CType> CTraits;

		// for every parameter: read the current value from the current state x
		typename CTraits::ParamVecType actualParams=boost::fusion::transform(constraint.getParams(),[&]<typename T>(RandomVariable<T> rv) //-> typename ManifoldTraits<T>::Manifold::Vec
		{
			int offs=this->getRVOffset(rv);
			constexpr int Dim=ManifoldTraits<T>::Manifold::Dim;
			//return x.segment<Dim>(offs);
			typename ManifoldTraits<T>::Manifold::Vec ret=x.segment<Dim>(offs);
			return ret;
		});

		typename CTraits::ParamJacType jac;

		auto jacptr=boost::fusion::transform(jac,[]<typename T>(T& t)
		{
			return (typename std::remove_const<T>::type*)&t;
		});


		auto fnParams=boost::fusion::push_front(boost::fusion::join(actualParams,jacptr),&constraint);
		typename CTraits::RetType residual=boost::fusion::invoke(&CType::error,fnParams);

		boost::mpl::for_each<boost::mpl::range_c<int,0,CTraits::NumParams> >([&]<typename Index1>(Index1 a)
		{
			(void)a;
			constexpr int Dim1=Index1::value;

			auto params=constraint.getParams();
			//std::cout << "Iterating" << T::value << std::endl;
			boost::mpl::for_each<boost::mpl::range_c<int,0,CTraits::NumParams> >([&]<typename Index2>(Index2 a)
			{
				(void)a;
				//std::cout << "Iterating" << Dim1 << " " << T::value << std::endl;
				constexpr int Dim1=Index1::value;
				constexpr int Dim2=Index2::value;

				typedef typename boost::fusion::result_of::value_at_c<typename CTraits::ParamJacType,Dim1>::type JacA;
				typedef typename boost::fusion::result_of::value_at_c<typename CTraits::ParamJacType,Dim2>::type JacB;

				Matrix<double,JacA::ColsAtCompileTime,JacB::ColsAtCompileTime> jtj=boost::fusion::at_c<Dim1>(jac).transpose()*boost::fusion::at_c<Dim2>(jac);

				const int aOffset=this->getRVDeltaOffset(boost::fusion::at_c<Dim1>(params));
				const int bOffset=this->getRVDeltaOffset(boost::fusion::at_c<Dim2>(params));

				for (int i=0;i<JacA::ColsAtCompileTime;i++)
						for (int j=0;j<JacB::ColsAtCompileTime;j++)
								sys.JTJ.coeffRef(aOffset+i,bOffset+j)+=jtj(i,j);
			});


			const int aRv=this->getRVDeltaOffset(boost::fusion::at_c<Dim1>(params));
			typedef typename boost::fusion::result_of::value_at_c<typename CTraits::ParamJacType,Dim1>::type JacA;

			sys.mJTz.segment<JacA::ColsAtCompileTime>(aRv)+=-boost::fusion::at_c<Dim1>(jac).transpose()*residual;

		});
	}
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
		boost::fusion::for_each(getConstraints(),[&]<typename CType>(const std::vector<CType>& constraints)
		{
			//       Meas
			//    ----------------
			// Rvs|
			//    |
			//    |
			for (const CType& constraint : constraints)
			{
				this->addConstraintToSystem(x, constraint,sys);
			}
		});

		//for (int i=0;i<6;i++)
		//sys.JTJ.coeffRef(i,i)+=1;
		//sys.JTJ.coeffRef(0,0)+=1;
		//sys.JTJ.coeffRef(1,1)+=1;
		//sys.JTJ.coeffRef(2,2)+=1;
		//std::cout << sys.JTJ << std::endl;
	}
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
		int numThreadsv=numThreads; // work around compiler bug
		ThreadPool* pools[numThreads];
		LinearSystem systems[numThreads];
		for (int i=0;i<numThreads;i++)
		{
			pools[i]=new ThreadPool(1);
			systems[i]=sys;
		}

		int task=0;
		boost::fusion::for_each(getConstraints(),[&]<typename CType>(const std::vector<CType>& constraints)
		{
			//       Meas
			//    ----------------
			// Rvs|
			//    |
			//    |
			for (const CType& constraint : constraints)
			{
				const int th=task%numThreadsv;

				pools[th]->pushTask([&,th]()
				{
					this->addConstraintToSystem(x, constraint,systems[th]);
				});
				task++;
			}
		});

		for (int i=0;i<numThreads;i++)
		{
			pools[i]->joinAllTasks();
			delete pools[i];
			sys.JTJ+=systems[i].JTJ;
			sys.mJTz+=systems[i].mJTz;
		}
	}
	void updateSystemSingleConstraint(const VectorXd& x, LinearSystem& sys)
	{
		sys.mJTz=VectorXd::Zero(getDeltaSize());

		boost::fusion::for_each(getConstraints(),[&]<typename CType>(const std::vector<CType>& constraints)
		{
			//       Meas
			//    ----------------
			// Rvs|
			//    |
			//    |
			if (constraints.empty())
				return;
			int i=rand()%constraints.size();
			{
				const CType& constraint=constraints[i];

				this->addConstraintToSystem(x, constraint,sys);
			}
		});
	}
	void updateDenseSystem(const VectorXd& x, LinearSystem& sys)
	{
		(void)x;
		(void)sys;
		assert(0 && "Not implemented");
	}
	//-----------------------------------------------------------------------------
	//-----------------------------------------------------------------------------
	VectorXd computeError(const VectorXd& x)
	{
		int measurementDim=0;
		boost::fusion::for_each(getConstraints(),[&]<typename T>(const std::vector<T>& vec)
		{
			measurementDim+=vec.size()*ConstraintTraits<T>::RetType::RowsAtCompileTime;
		});

		VectorXd err(measurementDim);
		int measOffs=0;
		boost::fusion::for_each(getConstraints(),[&]<typename CType>(const std::vector<CType>& constraints)
		{
			for (const CType& constraint : constraints)
			{
				typedef ConstraintTraits<CType> CTraits;
				typename CTraits::ParamVecType actualParams=boost::fusion::transform(constraint.getParams(),[&]<typename T>(RandomVariable<T> rv) //-> typename ManifoldTraits<T>::Manifold::Vec
				{
					int offs=this->getRVOffset(rv);
					constexpr int Dim=ManifoldTraits<T>::Manifold::Dim;
					//return x.segment<Dim>(offs);
					typename ManifoldTraits<T>::Manifold::Vec ret=x.segment<Dim>(offs);
					return ret;
				});

				typename CTraits::ParamJacType jac;
				auto jacptr=boost::fusion::transform(jac,[]<typename T>(T& t)
				{
					return (typename std::remove_const<T>::type*)nullptr;
				});

				//boost::mpl::print<decltype(jacptr)> dsdsads;
				auto fnParams=boost::fusion::push_front(boost::fusion::join(actualParams,jacptr),&constraint);
				//boost::mpl::print<decltype(fnParams)> asdiohsadisaod;
				typename CTraits::RetType residual=boost::fusion::invoke(&CType::error,fnParams);

				constexpr int DeltaDim=ManifoldTraits<typename CTraits::RetType>::Manifold::DeltaDim;

				err.segment<DeltaDim>(measOffs)=residual;
				measOffs+=DeltaDim;
			}

		});
		return err;
	}
	//-----------------------------------------------------------------------------
	//-----------------------------------------------------------------------------
	void generateNonZeroPattern(LinearSystem& sys)
	{
		int measurementDim=0;
		boost::fusion::for_each(getConstraints(),[&]<typename T>(const std::vector<T>& vec)
		{
			measurementDim+=vec.size()*ConstraintTraits<T>::RetType::RowsAtCompileTime;
		});

		std::vector<Eigen::Triplet<double> > triplets;
		int measOffset=0;
		boost::fusion::for_each(getConstraints(),[&]<typename CType>(const std::vector<CType>& constraints)
		{
			typedef ConstraintTraits<CType> CTraits;
			for (const CType& constraint : constraints)
			{
				// loop over all params of this measurement
				boost::fusion::for_each(constraint.getParams(),[&]<typename T>(RandomVariable<T> paramRv)
				{
					const int rvOffsFrom=this->getRVDeltaOffset(paramRv);

					constexpr int DeltaDim=CTraits::RetType::RowsAtCompileTime;
					for (int j=0;j<DeltaDim;j++)
						for (int i=0;i<RandomVariable<T>::ManifoldType::DeltaDim;i++)
							//triplets.push_back(Eigen::Triplet<double>{rvOffsFrom+i,measOffset+j,1});
							triplets.push_back({(rvOffsFrom+i),(measOffset+j),1});
				});

				constexpr int DeltaDim=CTraits::RetType::RowsAtCompileTime;
				measOffset+=DeltaDim;
			}
		});

		Eigen::SparseMatrix<double> HT{getDeltaSize(),measurementDim};
		HT.setFromTriplets(triplets.begin(),triplets.end());

		HT.finalize();
		sys.JTJ=HT*HT.transpose();

		//std::cout << HT << std::endl;
	}
	//-------------------------------------------------------------------------
	void finish(const VectorXd& x, /*const*/ Eigen::SparseMatrix<double>& cov)
	{
		boost::fusion::for_each(rvs,[&]<typename RVType>(const std::vector<RVType>& vec)
		{
			constexpr int Dim=RVType::ManifoldType::Dim;
			constexpr int DeltaDim=RVType::ManifoldType::DeltaDim;
			int offs1=boost::fusion::at_key<RVType>(this->offsets).stateOffs;
			int offs2=boost::fusion::at_key<RVType>(this->offsets).deltaOffs;
			for (int i=0;i<(int)vec.size();i++)
			{
				this->state(vec[i])=x.segment<Dim>(offs1);
				offs1+=Dim;
				offs2+=DeltaDim;
			}
		});
		(void)cov; // todo: use cov
	}
	void finishDense(const VectorXd& x, /*const*/ MatrixXd& cov)
	{
		boost::fusion::for_each(rvs,[&]<typename RVType>(const std::vector<RVType>& vec)
		{
			constexpr int Dim=RVType::ManifoldType::Dim;
			constexpr int DeltaDim=RVType::ManifoldType::DeltaDim;
			int offs1=boost::fusion::at_key<RVType>(this->offsets).stateOffs;
			int offs2=boost::fusion::at_key<RVType>(this->offsets).deltaOffs;
			for (int i=0;i<(int)vec.size();i++)
			{
				this->state(vec[i])=x.segment<Dim>(offs1);
				offs1+=Dim;
				offs2+=DeltaDim;
			}
		});
		(void)cov; // todo: use cov
	}
	//-------------------------------------------------------------------------
	void finishCov(const Eigen::MatrixXd& cov)
	{
		boost::fusion::for_each(rvs,[&]<typename RVType>(const std::vector<RVType>& vec)
		{
			constexpr int DeltaDim=RVType::ManifoldType::DeltaDim;
			int offs2=boost::fusion::at_key<RVType>(this->offsets).deltaOffs;
			for (int i=0;i<(int)vec.size();i++)
			{
				this->cov(vec[i])=cov.block<DeltaDim,DeltaDim>(offs2,offs2);
				offs2+=DeltaDim;
			}
		});
		(void)cov; // todo: use cov
	}

	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
	template<typename F>
	void forEachRV(F&& f) const
	{
		boost::fusion::for_each(rvs,[&]<typename RVType>(const std::vector<RVType>& vec)
		{
			for (auto& rv : vec)
				f(rv);
		});
	}
	template<typename RVType, typename F>
	void forEachRV(F&& f) const
	{
		auto& rvList=boost::fusion::at_key<std::vector<RandomVariable<RVType> > >(rvs);
		for (auto rv : rvList)
			f(rv);
	}
	template<typename F>
	void forEachConstraint(F&& f) const
	{
		boost::fusion::for_each(constraints,[&]<typename CType>(const std::vector<CType>& constraints)
		{
			for (auto& c : constraints)
				f(c);
		});
	}
	template<typename CType, typename F>
	void forEachConstraint(F&& f) const
	{
		const std::vector<CType>& constraintList=boost::fusion::at_key<std::vector<CType> >(constraints);
		for (auto c : constraintList)
			f(c);
	}
	//-------------------------------------------------------------------------
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
		boost::fusion::for_each(rvs,[&]<typename RVType>(const std::vector<RVType>& vec)
		{
			for (auto& rv : vec)
			{
				out << rv.getId() << "[label=\"" << typeid(RVType).name() << "_" << rv.getId() << "\"] ";
				//out << rv.getId() << "[label=\"" << rv.getId() << "\"] ";
			}
		});
		out << "\n\n";
		boost::fusion::for_each(constraints,[&]<typename CType>(const std::vector<CType>& constraints)
		{
			int cid=0;
			for (auto& c : constraints)
			{
				std::string cname=typeid(CType).name();
				out << "node [shape=box]; _" << cname << "_" << cid << "[label=\"" << cname <<  "\"]\n";
				//out << "node [shape=point]; _" << cname << "_" << cid << "[label=\"" << cname <<  "\"]\n";
				boost::fusion::for_each(c.getParams(),[&]<typename T>(RandomVariable<T> paramRv)
				{
					out << paramRv.getId() << " -- _" << cname << "_" << cid << "\n";
				});
				out << "\n";
				cid++;
			}
		});
		out << "}" << std::endl;
	}

};



//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

GSE_NS_END


#endif
#endif