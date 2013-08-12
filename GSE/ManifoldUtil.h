#ifndef GSE_MANIFOLD_UTIL_H
#define GSE_MANIFOLD_UTIL_H

#include"GSEDefs.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

GSE_NS_BEGIN

using namespace Eigen;


template<int N, typename Scalar_=double>
struct EuclideanManifold
{
	enum {Dim=N, DeltaDim=N };
	typedef Scalar_ Scalar;
	typedef Matrix<Scalar,Dim,1> Vec;
	typedef Matrix<Scalar,DeltaDim,1> DeltaVec;
	typedef Matrix<Scalar,DeltaDim,DeltaDim> DeltaMat;
	static Vec manifoldAdd(const Vec& x, const DeltaVec& d)
	{
		return x+d;
	}
	static DeltaVec manifoldSub(const Vec& x, const Vec& y)
	{
		return x-y;
	}
};

template<typename Scalar_=double>
struct SO3QuatManifold
{
	enum { Dim=4, DeltaDim=3 };
	typedef Scalar_ Scalar;
	typedef Matrix<Scalar, Dim, 1> Vec;
	typedef Matrix<Scalar, DeltaDim, 1> DeltaVec;
	typedef Matrix<Scalar, DeltaDim, DeltaDim> DeltaMat;

#define Tsegment template segment
	static Vec manifoldAdd(const Vec& x, const DeltaVec& d)
	{
		TQuat qx(x(0),x(1),x(2),x(3));
		TQuat qnew=qx*scaledAxisToQuat(d);

		Vec ret;
		ret(0)=qnew.w();	ret(1)=qnew.x();	ret(2)=qnew.y();	ret(3)=qnew.z();
		return ret;
	}

	static DeltaVec manifoldSub(const Vec& x, const Vec& y)
	{
		//RyTRx
		//RyT(x-y)
		TQuat qx(x(0),x(1),x(2),x(3));
		//TQuat qy(y(0),y(1),y(2),y(3));
		TQuat qy_conj(y(0),-1.0*y(1),-1.0*y(2),-1.0*y(3));
		TQuat qres=qy_conj*qx;

		return quatToScaledAxis(qres);
	}
private:
	typedef Eigen::Quaternion<Scalar> TQuat;
	typedef Matrix<Scalar,3,1> TVec3;

	static TQuat scaledAxisToQuat(const TVec3& axis)
	{
		Scalar angle=sqrt(axis(0)*axis(0)+axis(1)*axis(1)+axis(2)*axis(2)); // axis.norm() // axis.stableNorm()

		if(angle < Scalar(1e-10))
		{
			Scalar k1(1);
			Scalar k2(0.5*axis.x());
			Scalar k3(0.5*axis.y());
			Scalar k4(0.5*axis.z());
			Scalar norma=sqrt(k1*k1+k2*k2+k3*k3+k4*k4);
			return TQuat(k1/norma,k2/norma,k3/norma,k4/norma);
			//return TQuat(Scalar(1),Scalar(0),Scalar(0),Scalar(0));
		}

		TQuat qd;
		qd.w() = cos(0.5*angle);
		qd.vec()=axis*sin(0.5*angle)/angle;
		return qd;
	}

	static TVec3 quatToScaledAxis(const TQuat& q)
	{
		Scalar nv = sqrt(q.vec()(0)*q.vec()(0) + q.vec()(1)*q.vec()(1) + q.vec()(2)*q.vec()(2));
		if( nv < Scalar(1e-12))
		{
			nv = Scalar(1e-12);
			return 2.0*q.vec();
		}
		//nv+=1e-12;

		// BUGFIX 2009-11-30: q and -q are represent the same rotation
		//                    and must lead to the same result!
		// Note that singularity for w==0 is not dramatic, as
		// atan(+/-inf) = +/- pi/2 (in this case both solutions are acceptable)
		Scalar s = 2*std::atan(nv/(q.w()))/nv; // != 2*atan2(nv, w)/nv;
		//2.0*acos(std::min(1.0,q.w()))/nv;//

		return q.vec()*s;
	}
#undef Tsegment
};

typedef SO3QuatManifold<double> SO3QuatManifoldd;

template<typename Scalar_=double>
class SO2Manifold
{
public:
	enum {Dim=1, DeltaDim=1 };
	typedef Scalar_ Scalar;
	typedef Matrix<Scalar,Dim,1> Vec;
	typedef Matrix<Scalar,DeltaDim,1> DeltaVec;
	typedef Matrix<Scalar,DeltaDim,DeltaDim> DeltaMat;


#define Tsegment template segment
	static Vec manifoldAdd(const Vec& x, const DeltaVec& d)
	{
		return x+d;
	}
	static DeltaVec manifoldSub(const Vec& x, const Vec& y)
	{
		DeltaVec ret;
		Scalar delta=x(0)-y(0);
		ret(0)=myAtan2(sin(delta),cos(delta));
		return ret;
	}
private:
	static Scalar myAtan2(const Scalar& y, const Scalar& x)
	{
		/*return std::atan(y/x);*/
		return 2.0*std::atan(y/(sqrt(x*x+y*y)+x));
	}
#undef Tsegment

};

typedef SO2Manifold<double> SO2Manifoldd;

//-----------------------------------------------------------------------------

template <typename ...>
struct ProductManifold;

template<typename ManA, typename ... Mans>
struct ProductManifold<ManA, Mans ...>
{
	typedef ProductManifold<Mans ...> OtherMan;

	enum {Dim=ManA::Dim+OtherMan::Dim, DeltaDim=ManA::DeltaDim+OtherMan::DeltaDim };
	typedef Matrix<double,Dim,1> Vec;
	typedef Matrix<double,DeltaDim,1> DeltaVec;
	typedef Matrix<double,DeltaDim,DeltaDim> DeltaMat;

#define Tsegment template segment
	static Vec manifoldAdd(const Vec& x, const DeltaVec& d)
	{
		Vec ret;
		ret.Tsegment<ManA::Dim>(0)=ManA::manifoldAdd(x.Tsegment<ManA::Dim>(0),d.Tsegment<ManA::DeltaDim>(0));
		ret.Tsegment<OtherMan::Dim>(ManA::Dim)= OtherMan::manifoldAdd(x.Tsegment<OtherMan::Dim>(ManA::Dim),d.Tsegment<OtherMan::DeltaDim>(ManA::DeltaDim));
		return ret;
	}

	static DeltaVec manifoldSub(const Vec& x, const Vec& y)
	{
		DeltaVec ret;
		ret.Tsegment<ManA::DeltaDim>(0)=ManA::manifoldSub(x.Tsegment<ManA::Dim>(0),y.Tsegment<ManA::Dim>(0));
		ret.Tsegment<OtherMan::DeltaDim>(ManA::DeltaDim)=OtherMan::manifoldSub(x.Tsegment<OtherMan::Dim>(ManA::Dim),y.Tsegment<OtherMan::Dim>(ManA::Dim));
		return ret;
	}
#undef Tsegment
};

template <typename Man>
struct ProductManifold<Man> : Man
{ };

//-----------------------------------------------------------------------------

template<typename T>
struct ManifoldTraits
{
	typedef T Manifold;

	//typedef decltype(&T::manifoldAdd) CheckForManifoldAdd;
};
template<int N>
struct ManifoldTraits<Matrix<double,N,1> >
{
	typedef EuclideanManifold<N> Manifold;
};

template <typename T>
class IsManifold
{
	typedef char yes;
	typedef long no;
	template <typename C> static yes test( decltype(&C::manifoldAdd) ) ;
	template <typename C> static no test(...);
public:
	enum { value = sizeof(test<typename ManifoldTraits<T>::Manifold >(0)) == sizeof(char) };
};

template <typename T>
class IsNotManifold
{
	typedef char yes;
	typedef long no;
	template <typename C> static no test( decltype(&C::manifoldAdd) ) ;
	template <typename C> static yes test(...);
public:
	enum { value = sizeof(test<typename ManifoldTraits<T>::Manifold >(0)) == sizeof(char) };
};


GSE_NS_END

#endif
