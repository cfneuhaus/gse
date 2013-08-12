#ifndef GSE_RECBAYES_PARTICLEDISTRIBUTION_H
#define GSE_RECBAYES_PARTICLEDISTRIBUTION_H

#include "../GSEDefs.h"
#include <limits>
#include "../DiracDistribution.h"
#include "../Gaussian.h"

#include <Eigen/Dense>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/linear_congruential.hpp>


#include <boost/random/normal_distribution.hpp>
#include <memory>

GSE_NS_BEGIN

template <typename ParticleType>
struct Resampler
{
	virtual void resample(const std::vector<ParticleType*>& src, std::vector<ParticleType*>& dest) = 0;
};

template <typename ParticleType>
struct ImportanceResampling : public Resampler<ParticleType>
{
	virtual void resample(const std::vector<ParticleType*>& src, std::vector<ParticleType*>& dest)
	{
#if 1
		boost::rand48 randGen;
		boost::uniform_real<> randRange{0, 1};
		boost::variate_generator<boost::rand48&, boost::uniform_real<> > die{randGen, randRange};

		const double inverseNum = 1.0f / double(src.size());
		const double start = die() * inverseNum;  // random start in CDF
		unsigned int sourceIndex = 0;                     // index to draw from
		double cumulativeWeight = src[sourceIndex]->weight;
		for (unsigned int destIndex = 0; destIndex < dest.size(); destIndex++)
		{
			double probSum = start + inverseNum * destIndex;     // amount of cumulative weight to reach
			while (probSum > cumulativeWeight) // sum weights until
			{
				sourceIndex++;
				if (sourceIndex >= src.size())
				{
					sourceIndex = src.size() - 1;
					break;
				}
				cumulativeWeight += src[sourceIndex]->weight; // target sum reached
			}
			dest[destIndex]->state=src[sourceIndex]->state;
			dest[destIndex]->userdata=src[sourceIndex]->userdata;
			dest[destIndex]->weight=1.0;
		}
#else

		boost::rand48 randGen;
		std::vector<double> weights(src.size());
		for (int i=0;i<(int)src.size();++i)
			weights[i]=src[i]->weight;
		boost::random::discrete_distribution<> dist(weights);

		for (unsigned int destIndex = 0; destIndex < dest.size(); destIndex++)
		{
			int srcIndex=dist(gen);
			dest[destIndex]->state=src[srcIndex]->state;
			dest[destIndex]->weight=1.0;
		}
#endif
	}

};



struct NoUserData {};

template<typename StateVec, typename UserData=NoUserData>
struct Particle
{
	Particle() {}
	Particle(const StateVec& v, double w) : state(v), weight(w) {}
	//Particle(const StateVec& v, const UserData& userdata_, double w) : state(v), userdata(userdata_), weight(w) {}
	StateVec state;
	UserData userdata;
	double weight;
};


template <typename StateManifold, typename UserData=NoUserData, typename AugStateManifold=EuclideanManifold<0> >
class ParticleDistribution
{
public:
	//! Resampling modes.
	enum class ResamplingMode
	{
		/// never resample,
		RESAMPLE_NEVER,
		/// always resample
		RESAMPLE_ALWAYS,
		/// only resample if Neff < numParticles / 2
		RESAMPLE_NEFF
	};
private:
	typedef Matrix<double,StateManifold::Dim,1> StateVec;
	typedef Matrix<double,AugStateManifold::Dim,1> AugVec;

	typedef Particle<StateVec,UserData> ParticleType;

	unsigned int numParticles;

	//! A ParticleList is an array of pointers to Particles.
	typedef std::vector<ParticleType*> ParticleList;

	// Particle lists.
	// The particles are drawn from m_LastList to m_CurrentList to avoid new and delete commands.
	// In each run, the pointers m_CurrentList and m_LastList are switched in resample().
	ParticleList currentList;
	ParticleList lastList;

	// Stores a pointer to the resampling strategy.
	//ResamplingStrategy<StateType>* m_ResamplingStrategy;
	std::unique_ptr<Resampler<ParticleType> > resampler;

	// Stores which resampling mode is set, default is ResamplingMode::RESAMPLE_NEFF
	ResamplingMode resamplingMode;

public:
	template<typename SrcDist>
	ParticleDistribution(const SrcDist& srcDist, unsigned int numParticles_) : numParticles(numParticles_), resamplingMode(ResamplingMode::RESAMPLE_NEFF)
	{
		currentList.resize(numParticles);
		lastList.resize(numParticles);

		resampler.reset(new ImportanceResampling<ParticleType>);

		auto sampler=make_distSampler(srcDist);

		for (unsigned int i = 0; i < numParticles; i++)
		{
			currentList[i]=new ParticleType{sampler.sample(), 1.0/numParticles};
			lastList[i]=new ParticleType;
		}
	}

	ParticleDistribution& operator=(const ParticleDistribution& pd);

	~ParticleDistribution()
	{
		for (auto p : currentList)
			delete p;
		for (auto p : lastList)
			delete p;
	}

//	template<typename DestManifold, typename DestAugManifold, typename Functor>
//	AugmentedUnscentedDistribution<DestManifold,DestAugManifold,WeightingScheme> transform(Functor f)
//	{
//		AugmentedUnscentedDistribution<DestManifold,DestAugManifold,WeightingScheme> ret;
//		for (int i=0;i<WeightingScheme::NumSigmaPoints;i++)
//			ret.sigmapts.col(i)=f(sigmapts.col(i),augsigmapts.col(i));
//		if (DestAugManifold::Dim>0)
//			ret.augsigmapts=augsigmapts.template block<DestAugManifold::Dim,WeightingScheme::NumSigmaPoints>(augsigmapts.rows()-DestAugManifold::Dim,0);
//		ret.unscentedCombine();
//		return ret;
//	}

	//! @return Number of particles used in this filter
	unsigned int getNumParticles() const
	{
		return numParticles;
	}

	//! @param rs new resampling strategy
	//void setResamplingStrategy(ResamplingStrategy<StateType>* rs);

	//! @return the resampling strategy the particle filter currently uses
	//ResamplingStrategy<StateType>* getResamplingStrategy() const;

	//! Changes the resampling mode
	//! @param mode new resampling mode.
	//void setResamplingMode(ResamplingMode mode);

	//! @return the currently set resampling mode
	//ResamplingMode getResamplingMode() const;

	//! Computes and returns the number of effective particles.
	//! @return The estimated number of effective particles according to the formula:
	//!  \f[
	//!     N_{eff} = \frac{1}{\sum_{i=1}^{N_s} (w_k^i)^2}
	//!  \f]
	unsigned int getNumEffectiveParticles() const
	{
		double squareSum = 0;
		for (auto p : currentList)
			squareSum += p->weight * p->weight;
		return static_cast<int>(1.0f / squareSum);
	}

	/**
	 * Sets all particle states to the given state. Useful for integrating a
	 * known prior state to begin with tracking.
	 * @param priorState State that will be copied to all particles.
	 */
	//void setPriorState(const StateType& priorState);

	/**
	 * Draws all particle states from the given distribution.
	 * @param distribution The state distribution to draw the states from.
	 */
	//void drawAllFromDistribution(const StateDistribution<StateType>& distribution);

	const ParticleType& getBestParticle() const
	{
		return *currentList.front();
	}

	const StateVec& getBestState() const
	{
		return currentList.front()->state;
	}

	StateVec getBestXPercentEstimate(float percentage) const
	{
		const unsigned int numToConsider = numParticles * percentage;

		double totalWeight=0;
		for (unsigned int i = 0; i < numToConsider; i++)
			totalWeight+=currentList[i]->weight;

		StateVec mu=currentList[0]->state;
		for (int it=0;it<20;it++)
		{
			typename StateManifold::DeltaVec delta=StateManifold::DeltaVec::Zero();
			for (unsigned int i = 0; i < numToConsider; i++)
				delta+=(currentList[i]->weight/totalWeight)*StateManifold::manifoldSub(currentList[i]->state,mu);
			mu=StateManifold::manifoldAdd(mu,delta);
			if (delta.norm()<0.00001)
				break;
		}
		return mu;
	}

	//! This method selects a new set of particles out of an old set according to their weight
	//! (importance resampling). The particles from the list m_CurrentList points to are used as source,
	//! m_LastList points to the destination list. The pointers m_CurrentList and m_LastList are switched.
	//! The higher the weight of a particle, the more particles are drawn (copied) from this particle.
	//! The weight remains untouched, because measure() will be called afterwards.
	//! This method only works on a sorted m_CurrentList, therefore sort() is called first.
	void resample()
	{
		// swap lists
		currentList.swap(lastList);
		// call resampling strategy
		resampler->resample(lastList, currentList);
	}


	const ParticleList& getParticles() const { return currentList; }
	ParticleList& getParticles() { return currentList; }

	void update()
	{
		sort();
		normalize();
	}

	void checkResample()
	{
		if (resamplingMode == ResamplingMode::RESAMPLE_NEFF)
		{
			if (getNumEffectiveParticles() < getNumParticles() / 2)
				resample();
		}
		else if (resamplingMode == ResamplingMode::RESAMPLE_ALWAYS)
			resample();
	}

protected:

	void sort()
	{
		std::sort(currentList.begin(), currentList.end(), [](const ParticleType* a, const ParticleType* b) { return a->weight > b->weight; });
	}

	//! This method normalizes the weights of the particles. After calling this function, the sum of the weights of
	//! all particles in the current particle list equals 1.0.
	//! In this function the sum of all weights of the particles of the current particle list is computed and each
	//! weight of each particle is devided through this sum.
	void normalize()
	{
		double weightSum = 0.0;
		for (auto p : currentList)
			weightSum+=p->weight;

		// only normalize if weightSum is big enough to devide
		if (weightSum <= numParticles * std::numeric_limits<double>::epsilon())
		{
			std::cerr << "WARNING: ParticleFilter::normalize(): Particle weights *very* small!" << std::endl;
			return;
		}
		for (auto p : currentList)
			p->weight/=weightSum;
	}


};

GSE_NS_END

#endif
