// EventManager.cpp

#include "EventManager.h"

#define DUMP_SIZE 300

EventManager* EventManager::instance_ = 0;

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
EventManager* EventManager::Instance() {
  if (0 == instance_){
    instance_ = new EventManager();  
  }
  return instance_;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
event_ptr EventManager::newEvent(Model* creator, std::string group) {
  event_ptr ev(new Event(this, creator, group));
  return ev;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool EventManager::isValidSchema(event_ptr ev) {
  if (schemas_.find(creator_) != schemas_.end()) {
    if (schemas_[creator_].find(group_) != schemas.end()) {
      event_ptr primary = schemas_[creator_][group_];
      if (! ev->schemaWithin(primary)) {
        return false;
      } 
    }
  }
  return true;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void EventManager::addEvent(event_ptr ev) {
  if (! isValidSchema(ev)) {
    std::string msg;
    msg = "Group '" + group_ + "' with different schema already exists.";
    throw CycGroupDataMismatch(msg);
  }

  schemas_[creator_][group_] = ev;
  events_.push_back(ev);
  notifyBacks();
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void EventManager::notifyBacks() {
  if (events_.size() < DUMP_SIZE) {
    return;
  }

  std::list<EventBackend*>::iterator it;
  for(it = backs_.begin(); it != backs_.end(); it++) {
    try {
      *it->notify(events_);
    } catch (CycException err) {
      CLOG(LEV_ERROR) << "Backend '" << *it->name() << "failed write with err: "
                      << err.what();
    }
  }
  events_.clear();
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void EventManager::registerBackend(EventBackend b) {
  backs_.push_back(b);
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void EventManager::close() {
  std::list<EventBackend*>::iterator it;
  for(it = backs_.begin(); it != backs_.end(); it++) {
    try {
      *it->close();
    } catch (CycException err) {
      CLOG(LEV_ERROR) << "Backend '" << *it->name() << "failed to close with err: "
                      << err.what();
    }
  }
}
