#ifndef __CONNECTED_COMPONENTS_HPP
#define __CONNECTED_COMPONENTS_HPP

#include <vector>
#include <queue>
#include <unordered_map>
#include "src/simulation/coordinate.hpp"

/*!
    \namespace tools
    \brief Generic helper routines.
*/
namespace tools {

  //! A single disjoint connected component on an integer grid.
  /*! lowerInclusive..upperExclusive is the axis-aligned bounding box: the box
      covers cells [lower.d, upper.d) on each axis d. members holds every cell
      index (in the same encoding the caller supplies, see labelConnectedComponents).
      Ported conceptually from MUSIC region_multibox.cc compute_components_per_level(). */
  struct ConnectedComponent {
    Coordinate<int> lowerInclusive;
    Coordinate<int> upperExclusive;
    std::vector<Coordinate<int>> members;
  };

  namespace {
    //! 26-connected periodic flood-fill label of a flagged-cell set on a grid of side n.
    /*! \param flaggedCoords list of flagged cell coordinates (each component in [0,n))
        \param n            grid side length (the domain wraps periodically with period n)
        \return one ConnectedComponent per disjoint 26-connected cluster.

        Note on periodic wrap: a component that straddles the periodic boundary is
        reported with min/max cell indices on each axis, so its [lower,upper) box can
        appear to span the whole axis. This matches MUSIC's accepted limitation and is
        only a concern for clusters deliberately placed across the box edge. */
    inline std::vector<ConnectedComponent>
    labelConnectedComponents(const std::vector<Coordinate<int>> &flaggedCoords, int n) {
      std::vector<ConnectedComponent> components;
      if (flaggedCoords.empty())
        return components;

      auto key = [n](int x, int y, int z) -> long {
        return (static_cast<long>(x) * n + y) * n + z;
      };

      // 0 = unvisited flagged cell, 1 = visited. Only flagged cells live in the map.
      std::unordered_map<long, char> state;
      state.reserve(flaggedCoords.size() * 2);
      for (auto &c : flaggedCoords)
        state[key(c.x, c.y, c.z)] = 0;

      for (auto &seed : flaggedCoords) {
        long seedKey = key(seed.x, seed.y, seed.z);
        if (state[seedKey] != 0)
          continue;

        ConnectedComponent comp;
        comp.lowerInclusive = seed;
        comp.upperExclusive = Coordinate<int>(seed.x + 1, seed.y + 1, seed.z + 1);

        std::queue<Coordinate<int>> q;
        q.push(seed);
        state[seedKey] = 1;

        while (!q.empty()) {
          Coordinate<int> c = q.front();
          q.pop();

          comp.members.push_back(c);
          if (c.x < comp.lowerInclusive.x) comp.lowerInclusive.x = c.x;
          if (c.y < comp.lowerInclusive.y) comp.lowerInclusive.y = c.y;
          if (c.z < comp.lowerInclusive.z) comp.lowerInclusive.z = c.z;
          if (c.x + 1 > comp.upperExclusive.x) comp.upperExclusive.x = c.x + 1;
          if (c.y + 1 > comp.upperExclusive.y) comp.upperExclusive.y = c.y + 1;
          if (c.z + 1 > comp.upperExclusive.z) comp.upperExclusive.z = c.z + 1;

          for (int di = -1; di <= 1; ++di)
            for (int dj = -1; dj <= 1; ++dj)
              for (int dk = -1; dk <= 1; ++dk) {
                if (di == 0 && dj == 0 && dk == 0) continue;
                int ni = (c.x + n + di) % n;
                int nj = (c.y + n + dj) % n;
                int nk = (c.z + n + dk) % n;
                auto it = state.find(key(ni, nj, nk));
                if (it == state.end() || it->second != 0) continue;
                it->second = 1;
                q.push(Coordinate<int>(ni, nj, nk));
              }
        }
        components.push_back(std::move(comp));
      }
      return components;
    }
  }

}

#endif
