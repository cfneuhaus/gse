#ifndef GSE_LS_RANDOMVARIABLE_H
#define GSE_LS_RANDOMVARIABLE_H

#include "../GSEDefs.h"
#include "../ManifoldUtil.h"
#include <boost/fusion/mpl.hpp>

GSE_NS_BEGIN

template<typename ManifoldType_>
class RandomVariable
{
public:
	typedef typename ManifoldTraits<ManifoldType_>::Manifold ManifoldType;
	RandomVariable() : id(-1) {}
	explicit RandomVariable(int id_) : id(id_) {}
	int getId() const { return id; }
	bool operator<(const RandomVariable<ManifoldType_>& other) const
	{
		return id<other.id;
	}
	bool operator==(const RandomVariable<ManifoldType_>& other) const
	{
		return id==other.id;
	}
private:
	int id;
};

GSE_NS_END

// allow RandomVariables to be used in unordered_maps
namespace std
{
template <typename T>
class hash<GSE::RandomVariable<T> >{
public :
	size_t operator()(const GSE::RandomVariable<T>& x) const
	{
		return hash<int>()(x.getId());
	}
};
}

GSE_NS_BEGIN

template<typename T>
struct IsRandomVariable : public boost::mpl::false_
{
};
template<typename T>
struct IsRandomVariable<RandomVariable<T> > : public boost::mpl::true_
{
};

GSE_NS_END

#endif
