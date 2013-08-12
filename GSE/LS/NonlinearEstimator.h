#ifndef GSE_LS_NONLINEARESTIMATOR_H
#define GSE_LS_NONLINEARESTIMATOR_H

#include "../GSEDefs.h"
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/Cholesky>

//#define GSE_DEBUG_NONLINEARESTIMATOR

//#define GSE_NO_CHOLMOD

#ifndef GSE_NO_CHOLMOD
#include <cholmod.h>
#include <Eigen/CholmodSupport>
#endif

#ifdef GSE_DEBUG_NONLINEARESTIMATOR
#include <boost/date_time/posix_time/posix_time_types.hpp>
static double getCurrentTime()
{
	return boost::posix_time::microsec_clock::local_time().time_of_day().total_milliseconds();
}
#endif

GSE_NS_BEGIN

struct LinearSystem
{
	Eigen::SparseMatrix<double> JTJ;
	MatrixXd JTJd;
	VectorXd mJTz;
};

// Implements Nonlinear Least Squares solving
// Requires:
// - a StateRepresentation class inst which defines how the application state (for example a graph) is mapped to a vector
// - a Measurement class inst, which is essentially a functor that allows making a measurement at a certain state
template<typename Problem>
struct NonlinearMinimizer
{
	enum class TYPE
	{
		GAUSS_NEWTON=1,
		LEVENBERG=2, // diag+=lambda
		MARQUARDT=3  // diag*=(1+lambda)
	};
	TYPE type;
	bool verbose;
	NonlinearMinimizer(TYPE type_=TYPE::GAUSS_NEWTON) : type(type_), verbose(false) {}
	VectorXd x;
	LinearSystem sys;
	double lastErr;
	void setInitialGuess(const VectorXd& guess) { x=guess; }
	const VectorXd& getResult() const { return x; }


	void work(Problem& prob, int numIterations)
	{
		if (type==TYPE::GAUSS_NEWTON)
			work_gn(prob,numIterations);
		else
			work_levmar(prob,numIterations);
	}


	void work_levmar(Problem& prob, int numIterations)
	{
#ifdef GSE_DEBUG_NONLINEARESTIMATOR
		const double init0=getCurrentTime();
#endif
		lastErr=0;
		sys=LinearSystem{};
		prob.initializeSystem(sys);
#ifdef GSE_DEBUG_NONLINEARESTIMATOR
		if (verbose) std::cout << "Initialization took " << getCurrentTime()-init0 << " ms\n";
#endif


		// Systems < this size will be solved with dense instead of sparse matrices
		const int switchToDenseAt=0;

#ifdef GSE_NO_CHOLMOD
		Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > slu;
		slu.analyzePattern(sys.JTJ);
#else
		Eigen::CholmodDecomposition<Eigen::SparseMatrix<double> > slu;
		slu.setMode(CholmodAuto);
		slu.analyzePattern(sys.JTJ);
#endif

		const double eps1=1e-8;
		const double eps2=1e-9;
		const double tau=1e-3;

		if (sys.mJTz.rows()<switchToDenseAt)
			prob.updateDenseSystem(x, sys);
		else
			prob.updateSystem(x, sys);

		//std::cout << "System Size: " << sys.mJTz.rows() << std::endl;

		double nu=2;

		double maxDiag=-999999;
		for (int i=0;i<sys.JTJ.rows();i++)
			if (sys.JTJ.coeffRef(i,i)>maxDiag)
				maxDiag=sys.JTJ.coeffRef(i,i);
		double mu=tau*maxDiag;

		VectorXd originalDiag(sys.JTJ.rows());
		for (int i=0;i<sys.JTJ.rows();i++)
			originalDiag(i)=sys.JTJ.coeffRef(i,i);

		VectorXd errCur=prob.computeError(x);

		if (verbose) std::cout << "Error before optimization: " << errCur.transpose()*errCur << std::endl;

		int it=0;
		for (it=0;it<numIterations;it++)
		{
			if (type==TYPE::LEVENBERG)
			{
				for (int i=0;i<sys.JTJ.rows();i++)
					sys.JTJ.coeffRef(i,i)=originalDiag(i)+mu;
			}
			else if (type==TYPE::MARQUARDT)
			{
				for (int i=0;i<sys.JTJ.rows();i++)
					sys.JTJ.coeffRef(i,i)=originalDiag(i)*(1+mu);
			}

#ifdef GSE_DEBUG_NONLINEARESTIMATOR
			if (verbose) std::cout << "done in " << dt0 << " ms" << std::endl;
			const double start1=getCurrentTime();
#endif
			auto result=sys.mJTz;
			if (sys.mJTz.rows()>=switchToDenseAt)
			{
#if 0
				Eigen::SparseLLT<Eigen::SparseMatrix<double>,Cholmod > slu(sys.JTJ);//, Eigen::ColApproxMinimumDegree);
				if (!slu.succeeded())
				{
					std::cerr << "Decomposition failed" << std::endl;
					break;
				}
				slu.solveInPlace(result);
				std::cout << "Rows: " << sys.mJTz.rows() << std::endl;
#else
				//Eigen::SimplicialLLt<Eigen::SparseMatrix<double> > slu(sys.JTJ);
				//slu.factorize(sys.JTJ);
				slu.compute(sys.JTJ);
				//slu.factorize(sys.JTJ);
				if(slu.info()!=Success)
				{
					std::cerr << "Factorization failed!" << std::endl;
					break;
				}
				auto tmp=result;
				tmp=slu.solve(result);
				if(slu.info()!=Success)
				{
					std::cerr << "Solving failed!" << std::endl;
					break;
				}
				result=tmp;
#endif
			}
			else
			{
				result=sys.JTJd.llt().solve(sys.mJTz);
			}
#ifdef GSE_DEBUG_NONLINEARESTIMATOR
			const double dt1=getCurrentTime()-start1;
			if (verbose) std::cout << "Solving took " << dt1 << " ms" << std::endl;
#endif

			if (std::isnan((double)(result.transpose()*result)))
			{
				std::cerr << "Divergence" << std::endl;
				return;
			}

			if (result.norm()<eps2*(x.norm()+eps2))
			{
				if (verbose) std::cout << "Eps2 condition reached" << std::endl;
				it++;
				break;
			}


			auto xnew=x;
			prob.addDelta(xnew,result);

			const VectorXd errNew=prob.computeError(xnew);

			const double Fx=0.5*errCur.transpose()*errCur;
			const double Fxnew=0.5*errNew.transpose()*errNew;
			const double denom=0.5*result.transpose()*(mu*result+sys.mJTz);
			const double gain=(Fx-Fxnew)/(denom);

			std::cout << "LevMar: Gain: " << gain;

			if (gain>0)
			{
				x=xnew;
				//mu/=10;
				mu=mu*std::max(1.0/3.0,1.0-pow(2*gain-1,3));
				nu=2;

				if (verbose) std::cout << " -> Accept | mu: " << mu << std::endl;

				if (sys.mJTz.rows()<switchToDenseAt)
					prob.updateDenseSystem(x, sys);
				else
					prob.updateSystem(x, sys);

				for (int i=0;i<sys.JTJ.rows();i++)
					originalDiag(i)=sys.JTJ.coeffRef(i,i);

				errCur=errNew;



				const double errSqr=errCur.transpose()*errCur;
				lastErr=errSqr;
				if (verbose)
				{
					std::cout << "Delta: Inf: " << result.array().abs().maxCoeff() << " Eukl: " << result.norm() << " ";
					std::cout << "Error: Inf: " << errNew.array().abs().maxCoeff() << " Sqr: " << errSqr << std::endl;
#ifdef GSE_DEBUG_NONLINEARESTIMATOR
					std::cout << "Iteration " << it+1 << "/" << numIterations << " done in " << getCurrentTime()-start0 << " ms!\n" << std::endl;
#else
					std::cout << "Iteration " << it+1 << "/" << numIterations << " done" << std::endl;
#endif
				}

				if (sys.mJTz.lpNorm<Infinity>()<eps1)
				{
					if (verbose) std::cout << "Eps1 condition reached" << std::endl;
					it++;
					break;
				}

			}
			else
			{
				//mu*=10;
				mu*=nu;
				nu*=2;
				if (verbose) std::cout << " -> Backoff | mu: " << mu << std::endl;
			}
		}

		const double errSqr=errCur.transpose()*errCur;
		if (verbose) std::cout << "Converged to ErrSqr: " << errSqr << " after " << it << " iterations " << std::endl;

	}





	void work_gn(Problem& prob, int numIterations)
	{
#ifdef GSE_DEBUG_NONLINEARESTIMATOR
		const double init0=getCurrentTime();
#endif
		//VectorXd x=t.getInitialGuess();
		lastErr=0;
		//LinearSystem sys;
		sys=LinearSystem{};
	prob.initializeSystem(sys);
#ifdef GSE_DEBUG_NONLINEARESTIMATOR
	if (verbose) std::cout << "Initialization took " << getCurrentTime()-init0 << " ms\n";
#endif

	// for levmar only
	double nu=2;
	double mu=1e-6;
	bool backedOff=false;

	double lastErrSqr=0;

	// compute error before:
	VectorXd errBefore=prob.computeError(x);
	if (verbose) std::cout << "Error before optimization: " << errBefore.transpose()*errBefore << std::endl;

	// Systems < this size will be solved with dense instead of sparse matrices
	const int switchToDenseAt=0;

#ifdef GSE_NO_CHOLMOD
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > slu;
	slu.analyzePattern(sys.JTJ);
#else
	Eigen::CholmodDecomposition<Eigen::SparseMatrix<double> > slu;
	slu.setMode(CholmodAuto);
	slu.analyzePattern(sys.JTJ);
#endif


	for (int it=0;it<numIterations;it++)
	{
		if (verbose) std::cout << "Updating system..." << std::flush;
#ifdef GSE_DEBUG_NONLINEARESTIMATOR
		const double start0=getCurrentTime();
#endif

		if (!backedOff)
		{
			if (sys.mJTz.rows()<switchToDenseAt)
				prob.updateDenseSystem(x, sys);
			else
				prob.updateSystem(x, sys);
		}
		backedOff=false;

#ifdef GSE_DEBUG_NONLINEARESTIMATOR
		const double dt0=getCurrentTime()-start0;
#endif

		//std::cout << "System Size: " << sys.mJTz.rows() << std::endl;

		if ((type==TYPE::LEVENBERG)||(type==TYPE::MARQUARDT))
		{
			if (it==0)
			{
				double maxDiag=-999999;
				for (int i=0;i<sys.JTJ.rows();i++)
					if (sys.JTJ.coeffRef(i,i)>maxDiag)
						maxDiag=sys.JTJ.coeffRef(i,i);
				mu=1e-6*maxDiag;
			}

			if (type==TYPE::LEVENBERG)
			{
				for (int i=0;i<sys.JTJ.rows();i++)
					sys.JTJ.coeffRef(i,i)+=mu;
			}
			else
			{
				for (int i=0;i<sys.JTJ.rows();i++)
					sys.JTJ.coeffRef(i,i)*=(1+mu);
			}
		}
#ifdef GSE_DEBUG_NONLINEARESTIMATOR
		if (verbose) std::cout << "done in " << dt0 << " ms" << std::endl;
		const double start1=getCurrentTime();
#endif
		auto result=sys.mJTz;
		if (sys.mJTz.rows()>=switchToDenseAt)
		{
#if 0
			Eigen::SparseLLT<Eigen::SparseMatrix<double>,Cholmod > slu(sys.JTJ);//, Eigen::ColApproxMinimumDegree);
			if (!slu.succeeded())
			{
				std::cerr << "Decomposition failed" << std::endl;
				break;
			}
			slu.solveInPlace(result);
			std::cout << "Rows: " << sys.mJTz.rows() << std::endl;
#else
			//Eigen::SimplicialLLt<Eigen::SparseMatrix<double> > slu(sys.JTJ);
			//slu.factorize(sys.JTJ);
			slu.compute(sys.JTJ);
			//slu.factorize(sys.JTJ);
			if(slu.info()!=Success)
			{
				std::cerr << "Factorization failed!" << std::endl;
				break;
			}
			auto tmp=result;
			tmp=slu.solve(result);
			if(slu.info()!=Success)
			{
				std::cerr << "Solving failed!" << std::endl;
				break;
			}
			result=tmp;
#endif
		}
		else
		{
			result=sys.JTJd.llt().solve(sys.mJTz);
		}
#ifdef GSE_DEBUG_NONLINEARESTIMATOR
		const double dt1=getCurrentTime()-start1;
		if (verbose) std::cout << "Solving took " << dt1 << " ms" << std::endl;
#endif


		auto xnew=x;
		prob.addDelta(xnew,result);

		VectorXd errNew=prob.computeError(xnew);

		if (type==TYPE::GAUSS_NEWTON)
			x=xnew;
		else
		{
			VectorXd errOld=prob.computeError(x);

			double Fx=0.5*errOld.transpose()*errOld;
			double Fxnew=0.5*errNew.transpose()*errNew;
			double denom=0.5*result.transpose()*(mu*result+sys.mJTz);
			double gain=(Fx-Fxnew)/(denom);

			std::cout << "LevMar: Gain: " << gain;

			if (gain>0)
			{
				x=xnew;
				mu=mu*10;//std::max(1.0/3.0,1.0-pow(2*gain-1,3));
				nu=2;

				std::cout << " -> Accept " << mu << " " << Fxnew << std::endl;
			}
			else
			{
				//mu*=nu;
				mu/=10;
				nu*=2;
				std::cout << " -> Backoff " << mu << " " << Fx << std::endl;
				backedOff=true; // dont recompute jacobian
			}
		}

		//VectorXd errOld=meas.computeError(x);
		//std::cout << "OLD ERR: " << 0.5*errOld.transpose()*errOld << std::endl;

		const double errSqr=errNew.transpose()*errNew;
		lastErr=errSqr;
		if (verbose)
		{
			std::cout << "Delta: Inf: " << result.array().abs().maxCoeff() << " Eukl: " << result.norm() << " ";
			std::cout << "Error: Inf: " << errNew.array().abs().maxCoeff() << " Sqr: " << errSqr << std::endl;
#ifdef GSE_DEBUG_NONLINEARESTIMATOR
			std::cout << "Iteration " << it+1 << "/" << numIterations << " done in " << getCurrentTime()-start0 << " ms!\n" << std::endl;
#else
			std::cout << "Iteration " << it+1 << "/" << numIterations << " done" << std::endl;
#endif
		}

		if (std::isnan((double)(result.transpose()*result)))
		{
			std::cerr << "Divergence" << std::endl;
			break;
		}

		if (!backedOff)
		{
			if (fabs(errSqr-lastErrSqr)<1e-4)//result.transpose()*result
			{
				if (verbose) std::cout << "Converged to ErrSqr: " << errSqr << " after " << it+1 << " iterations " << std::endl;
				break;
			}
			lastErrSqr=errSqr;
		}
	}

	//if (sys.mJTz.rows()>=switchToDenseAt)
	//	t.finish(x,sys.JTJ);
	//	else
	//			t.finishDense(x,sys.JTJd);
}

};

GSE_NS_END

#endif // NONLINEARESTIMATOR_H
