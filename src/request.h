#ifndef CYCLUS_REQUEST_H_
#define CYCLUS_REQUEST_H_

#include <ostream>
#include <string>

#include <boost/shared_ptr.hpp>

#include "cyc_limits.h"

namespace cyclus {

class Trader;
template <class T> class RequestPortfolio;

/// @class Request
///
/// @brief A Request encapsulates all the information required to communicate
/// the needs of an agent in the Dynamic Resource Exchange, including the
/// commodity it needs as well as a resource specification for that commodity.
/// A Request is templated its resource.
template <class T>
class Request {
 public:
  typedef boost::shared_ptr< Request<T> > Ptr;

  Request(boost::shared_ptr<T> target, Trader* requester,
          std::string commodity = "", double preference = 0)
    : target_(target),
      requester_(requester),
      commodity_(commodity),
      preference_(preference),
      id_(next_id_++) {};

  /// @return this request's target
  inline boost::shared_ptr<T> target() const {
    return target_;
  }

  /// @return the requester associated with this request
  inline Trader* requester() const {
    return requester_;
  }
  
  /// @return the commodity associated with this request
  inline std::string commodity() const {
    return commodity_;
  }

  /// @return the preference value for this request
  inline double preference() const {
    return preference_;
  }
  
  /// @return a unique id for the request
  inline int id() const {
    return id_;
  }

  /// @brief set the portfolio for this request
  inline void set_portfolio(typename RequestPortfolio<T>::Ptr portfolio) {
    portfolio_ = portfolio;
  }

  /// @return the portfolio of which this request is a part
  inline typename RequestPortfolio<T>::Ptr portfolio() const {
    return portfolio_;
  }

  boost::shared_ptr<T> target_;
  Trader* requester_;
  double preference_;
  std::string commodity_;
  typename RequestPortfolio<T>::Ptr portfolio_;
  int id_;
  static int next_id_;
};

template<class T> int Request<T>::next_id_ = 0;

/// @brief Request-Request equality operator
template<class T>
inline bool operator==(const Request<T>& lhs,
                       const Request<T>& rhs) {
  return (lhs.commodity() == rhs.commodity() &&
          lhs.target() == rhs.target() &&
          lhs.portfolio() == rhs.portfolio() &&
          AlmostEq(lhs.preference(), rhs.preference()) &&
          lhs.requester() == rhs.requester());
}

/// @brief Request-Request comparison operator, allows usage in ordered containers
template<class T>
inline bool operator<(const cyclus::Request<T>& lhs,
                      const cyclus::Request<T>& rhs) {
  return  (lhs.id() < rhs.id());
};

} // namespace cyclus

#endif
