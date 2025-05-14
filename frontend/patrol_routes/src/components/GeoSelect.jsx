import { useState } from "react"
import { features } from "../../../../geodata/comunas.json"

function GeoSelect({features, selIndex, setSelIndex, name='', idInFeature=false}){
  return (
    <div className="form-group">
      <label htmlFor={name}>{name}</label>
      <select
        id={name}
        name={name}
        value={selIndex}
        onChange={e => setSelIndex(e.target.value)}
        className="form-control">
        {idInFeature
        ? features.map((feature, i) =>
          <option key={i} value={feature.properties.id}>
            {feature.properties.name}
          </option>)
        : features.map((feature, i) =>
          <option key={i} value={i}>{feature.properties.name}</option>
        )}
      </select>
    </div>
  )
}

export default GeoSelect;
