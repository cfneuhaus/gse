#ifndef GSE_RECBAYES_UKF_H
#define GSE_RECBAYES_UKF_H

#include "../GSEDefs.h"
#include <Eigen/Core>
//#include "Unscented.h"
#include <iostream>
#include <Eigen/Cholesky>
#include <functional>
#include "../ManifoldUtil.h"
#include "../Gaussian.h"

GSE_NS_BEGIN

using namespace Eigen;

template<int N>
struct BlockDiagonalInitializer
{
	BlockDiagonalInitializer(Matrix<double,N,N>& m_) : m(m_), i(0) { m.setZero(); }

	template<int NB>
	BlockDiagonalInitializer& add(const Matrix<double,NB,NB>& b)
	{
		m.template block<NB,NB>(i,i)=b;
		i+=NB;
		return *this;
	}
	template<typename M>
	BlockDiagonalInitializer& add(const M& b)
	{
		assert(b.rows()==b.cols());
		m.template block(i,i,b.rows(),b.cols())=b;
		i+=b.rows();
		return *this;
	}
private:
	Matrix<double,N,N>& m;
	int i;
};

template<int N>
BlockDiagonalInitializer<N> blockDiagonalInit(Matrix<double,N,N>& m) { return {m}; }



template<typename Distribution>
class UKF
{
public:
	UKF(Distribution& distrib_) : distrib(distrib_) {}

	template<typename SysModel, typename ObsModel, typename Observer>
	void predict_update(SysModel& sysModel, const double dt, ObsModel& obsModel, Observer& obs);

	template<typename SysModel>
	void predict(SysModel& sysModel, const double dt);

	template<typename ObsModel, typename Observer>
	void update(ObsModel& obsModel, Observer& obs);

private:
	Distribution& distrib;
};

//-----------------------------------------------------------------------------
template<typename Distribution> template<typename SysModel>
void UKF<Distribution>::predict(SysModel& sysModel, const double dt)
{
	typedef Matrix<double,SysModel::StateDim,1> StateVec;
	typedef Matrix<double,SysModel::SystemNoiseDim,1> NoiseVec;

	auto ud=make_augmentedUnscentedDistribution(distrib,make_sqrtGaussian(make_euclideanGaussian(sysModel.getNoise(dt))));

	auto augPred=ud.template transform<typename SysModel::ManifoldType>([&sysModel,dt](const StateVec& chi, const NoiseVec& noise)
	{
		return sysModel.advanceState(chi,noise,dt);
	});
	//using namespace std::placeholders;
	//auto augPred=ud.template transform<typename SysModel::ManifoldType>(std::bind(&SysModel::advanceState,sysModel,_1,_2,dt));

	distrib=augPred.toSqrtGaussian();
}
//-----------------------------------------------------------------------------
template<typename Distribution> template<typename ObsModel, typename Observer>
void UKF<Distribution>::update(ObsModel& obsModel, Observer& obs)
{
	typedef Matrix<double,ObsModel::StateDim,1> StateVec;
	typedef Matrix<double,ObsModel::ObsDim,1> ObsVec;
	typedef typename ObsModel::ObsNoiseVec ObsNoiseVec;

	auto ud=make_augmentedUnscentedDistribution(distrib,make_sqrtGaussian(make_euclideanGaussian(obs.getNoise())));

	auto predMeas=ud.template transform<typename ObsModel::ManifoldType>([&obsModel](const StateVec& chi, const ObsNoiseVec& obsNoise)
	{
		return obsModel.augmentedPredictMeasurement(chi,obsNoise);
	});
	//using namespace std::placeholders;
	//auto predMeas=ud.template transform<typename ObsModel::ManifoldType>(std::bind(&ObsModel::augmentedPredictMeasurement,obsModel,_1,_2));


	Matrix<double,Distribution::Manifold::DeltaDim,ObsModel::ObsDeltaDim> cross=crossCov(ud,predMeas);

	auto predMeasGauss=predMeas.toSqrtGaussian();

	ObsVec m;
	if (!obs.measure(predMeasGauss.toGaussian(),m)) // a bit hacky
	{
		std::cout << "Measurement failed! Skipping Kalman Update" << std::endl;
		return;
	}

	auto sqrtS=predMeasGauss.getCovL();

	// K=Cross*(S*S^T)^-1
	// K*S*S^T=Cross
	// S*S^T*K^T=Cross^T
	// S*Y=Cross^T
	// S^T*K^T=Y
	//
	// Bei EKF: Cross=P*H^T
	Matrix<double,ObsModel::ObsDeltaDim,Distribution::Manifold::DeltaDim> KT=cross.transpose();
	sqrtS.solveInPlace(KT);
	sqrtS.transpose().solveInPlace(KT);

	Matrix<double,Distribution::Manifold::DeltaDim,ObsModel::ObsDeltaDim> U=KT.transpose()*sqrtS;
	distrib.addDelta(KT.transpose()*ObsModel::ManifoldType::manifoldSub(m,predMeasGauss.mu),U);
}
//-----------------------------------------------------------------------------
template<typename Distribution> template<typename SysModel, typename ObsModel, typename Observer>
void UKF<Distribution>::predict_update(SysModel& sysModel, const double dt, ObsModel& obsModel, Observer &obs)
{
	typedef Matrix<double,SysModel::StateDim,1> StateVec;

	typedef EuclideanManifold<SysModel::SystemNoiseDim+ObsModel::ObsNoiseDim> NoiseManifold;
	typedef typename NoiseManifold::Vec NoiseVec;
	typedef EuclideanManifold<ObsModel::ObsNoiseDim> ObsNoiseManifold;

	SqrtGaussian<NoiseManifold> noise;
	noise.mu=NoiseVec::Zero();

	blockDiagonalInit(noise.covL)
			.add(sysModel.getNoise(dt).llt().matrixL()) // a bit hacky
			.add(obs.getNoise().llt().matrixL()); // a bit hacky

	auto ud=make_augmentedUnscentedDistribution(distrib,noise);

    auto augPred=ud.template transform<typename SysModel::ManifoldType,ObsNoiseManifold>([&sysModel,dt](const StateVec& chi, const NoiseVec& noise) -> StateVec
	{
		typedef Matrix<double,SysModel::SystemNoiseDim,1> SystemNoiseVec;

		SystemNoiseVec systemNoise=noise.template segment<SysModel::SystemNoiseDim>(0);
		return sysModel.advanceState(chi,systemNoise,dt);
	});

	distrib=augPred.toSqrtGaussian();

	//////////////////////////////////////// UPDATE

	typedef Matrix<double,ObsModel::ObsDim,1> ObsVec;
	typedef typename ObsModel::ObsNoiseVec ObsNoiseVec;

	auto predMeas=augPred.template transform<typename ObsModel::ManifoldType>([&obsModel](const StateVec& chi, const ObsNoiseVec& obsNoise)
	{
		return obsModel.augmentedPredictMeasurement(chi,obsNoise);
	});

	Matrix<double,SysModel::StateDeltaDim,ObsModel::ObsDeltaDim> cross=crossCov(augPred,predMeas);

	auto predMeasGauss=predMeas.toSqrtGaussian();

	ObsVec m;
	if (!obs.measure(predMeasGauss.toGaussian(),m)) // a bit hacky
	{
		std::cout << "Measurement failed! Skipping Kalman Update" << std::endl;
		return;
	}

	auto sqrtS=predMeasGauss.getCovL();

	// K=Cross*(S*S^T)^-1
	// K*S*S^T=Cross
	// S*S^T*K^T=Cross^T
	// S*Y=Cross^T
	// S^T*K^T=Y
	//
	// Bei EKF: Cross=P*H^T
	Matrix<double,ObsModel::ObsDeltaDim,SysModel::StateDeltaDim> KT=cross.transpose();
	sqrtS.solveInPlace(KT);
	sqrtS.transpose().solveInPlace(KT);

	Matrix<double,SysModel::StateDeltaDim,ObsModel::ObsDeltaDim> U=KT.transpose()*sqrtS;
	distrib.addDelta(KT.transpose()*ObsModel::ManifoldType::manifoldSub(m,predMeasGauss.mu),U);
}


template<typename Dist>
UKF<Dist> make_ukf(Dist& pd) { return {pd}; }

GSE_NS_END

#endif
