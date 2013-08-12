#include "FitLineSample.h"
#include <GSE/LS/LSProblem.h>
#include <GSE/LS/WrappedConstraint.h>
#include <GSE/LS/NumDiff.h>

using namespace GSE;


void fitLine()
{
	class MyLineConstraint
	{
	public:
		typedef boost::fusion::vector<RandomVariable<Vector2d> > ParamRVs;

		MyLineConstraint(ParamRVs params_, const Vector2d& pt_) : params(params_), pt(pt_) {}

		Matrix<double,1,1> error(const Vector2d& x, Matrix<double,1,2>* dx) const
		{
			Matrix<double,1,1> err;
			err << pt(1)-x(0)*pt(0)+x(1);
			if (dx)
			{
				(*dx) << -pt(0), 1;
				// alternatively:
				//auto jac=computeJacobian(*this,boost::fusion::make_vector(x));
				//(*dx)=boost::fusion::at_c<0>(jac);
			}
			return err;
		}

		const ParamRVs& getParams() const { return params; }
	private:
		ParamRVs params;
		Vector2d pt;
	};

	typedef LSProblem<Vector2d,MyLineConstraint> LineFitProblem;
	LineFitProblem g;

	auto rvline=g.addRandomVariable<Vector2d>();

	g.addConstraint(MyLineConstraint(rvline,Vector2d(0,0)));
	g.addConstraint(MyLineConstraint(rvline,Vector2d(5,5)));
	g.addConstraint(MyLineConstraint(rvline,Vector2d(3,4)));

	g.optimize();

	std::cout << "Result: y=ax+b ; (a,b)=" << g.state(rvline).transpose() << std::endl;
}
